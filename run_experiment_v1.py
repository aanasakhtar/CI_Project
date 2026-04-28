"""
run_experiment.py
═════════════════
Runs both the baseline HP-MOCD and the overlapping extension on both
datasets (LFR and DBLP) and prints a comparison table.

Place this file in the project root (same level as baseline/ and data/).

Usage
-----
    python run_experiment.py                  # both datasets, both methods
    python run_experiment.py --dataset lfr    # LFR only
    python run_experiment.py --dataset dblp   # DBLP only
"""

import sys
import argparse
from pathlib import Path
import numpy as np


# ── Make sure project root is on the path ────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import networkx as nx

from HPMOCD.hp_mocd_baseline       import run_minimal_nsgaii
from HPMOCD.hp_mocd_overlapping    import run_hp_mocd_overlapping
from evaluation.metrics              import evaluate_disjoint, evaluate_overlapping
from evaluation.partition_utils      import save_partition_report


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _print_scores(label: str, scores: dict[str, float]) -> None:
    """Print a metric dict in a readable table row."""
    parts = [f"{k}={v:.4f}" for k, v in scores.items()]
    print(f"  [{label}]  " + "  |  ".join(parts))


def _slugify(label: str) -> str:
    """Create a filesystem-safe slug for output filenames."""
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in label).strip("_")


def _print_membership_preview(
    label: str,
    node_to_communities: dict[int, list[int]],
    max_nodes: int,
) -> None:
    """Print a compact preview of node memberships for quick inspection."""
    items = list(node_to_communities.items())
    shown = items[:max_nodes]

    preview = ", ".join(f"{node}:{memberships}" for node, memberships in shown)
    print(
        f"  [{label}] node->communities preview "
        f"({len(shown)}/{len(items)} nodes): {preview}"
    )


def _run_disjoint_lfr(
    output_dir: Path,
    preview_memberships: int,
) -> list[frozenset]:
    """Run the stronger local NSGA-II baseline on the disjoint LFR benchmark."""
    print("\nLoading disjoint LFR benchmark ...")
    from data.load_lfr import load_lfr_disjoint

    G, ground_truth = load_lfr_disjoint()
    print(f"\n{'='*65}")
    print("  DATASET: LFR Disjoint Benchmark")
    print(f"  Nodes={G.number_of_nodes()}  Edges={G.number_of_edges()}")
    print(f"  Ground-truth communities: {len(ground_truth)}")
    print(f"{'='*65}")

    print("\n>> Running stronger MinimalNSGAII baseline ...")
    baseline_partition, baseline_rt = run_minimal_nsgaii(G)
    print(f"   Detected {len(baseline_partition)} communities in {baseline_rt:.2f}s")

    baseline_scores = evaluate_disjoint(G, baseline_partition, ground_truth)
    _print_scores("Disjoint baseline", baseline_scores)

    baseline_output = output_dir / "lfr_disjoint_baseline_memberships.json"
    baseline_node_map = save_partition_report(
        baseline_output,
        "LFR Disjoint Benchmark",
        "minimal_nsgaii",
        baseline_partition,
    )
    _print_membership_preview("Disjoint baseline", baseline_node_map, preview_memberships)
    print(f"   Saved baseline memberships: {baseline_output}")
    return baseline_partition


def _run_overlap_lfr(
    output_dir: Path,
    preview_memberships: int,
    overlapping_n_communities: int | None,
    seed_partition: list[frozenset] | None = None,
) -> None:
    """Run the overlapping extension on the overlapping LFR benchmark."""
    print("\nLoading overlapping LFR benchmark ...")
    from data.load_lfr import load_lfr_overlapping

    G, ground_truth = load_lfr_overlapping()
    print(f"\n{'='*65}")
    print("  DATASET: LFR Overlapping Benchmark")
    print(f"  Nodes={G.number_of_nodes()}  Edges={G.number_of_edges()}")
    print(f"  Ground-truth communities: {len(ground_truth)}")
    print(f"{'='*65}")

    overlap_init = (
        max(2, overlapping_n_communities)
        if overlapping_n_communities is not None
        else max(2, int(len(G) ** 0.5 // 2))
    )
    print(
        "  [DIAG] Overlapping initial labels="
        f"{overlap_init}; this is an initial label pool (not a hard cap)."
    )

    print("\n>> Running overlapping HP-MOCD-like extension ...")
    overlapping_partition, overlapping_rt = run_hp_mocd_overlapping(
        G,
        max_memberships=2,
        n_communities=overlapping_n_communities,
        seed_partition=seed_partition,
    )
    print(f"   Detected {len(overlapping_partition)} communities in {overlapping_rt:.2f}s")

    node_counts: dict[int, int] = {}
    for community in overlapping_partition:
        for node in community:
            node_counts[node] = node_counts.get(node, 0) + 1
    n_overlapping = sum(1 for c in node_counts.values() if c > 1)
    print(f"   Overlapping nodes: {n_overlapping} / {G.number_of_nodes()}")

    overlapping_scores = evaluate_overlapping(G, overlapping_partition, ground_truth)
    _print_scores("Overlapping extension", overlapping_scores)

    overlapping_output = output_dir / "lfr_overlapping_extension_memberships.json"
    overlapping_node_map = save_partition_report(
        overlapping_output,
        "LFR Overlapping Benchmark",
        "overlapping_extension",
        overlapping_partition,
    )
    _print_membership_preview("Overlapping extension", overlapping_node_map, preview_memberships)
    print(f"   Saved overlapping memberships: {overlapping_output}")


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET LOADERS
# ══════════════════════════════════════════════════════════════════════════════

def run_lfr(
    output_dir: Path,
    preview_memberships: int,
    overlapping_n_communities: int | None,
) -> None:
    """Run the disjoint phase first, then the overlapping phase."""
    seed_partition = _run_disjoint_lfr(output_dir, preview_memberships)
    _run_overlap_lfr(output_dir, preview_memberships, overlapping_n_communities, seed_partition)


def run_dblp(
    output_dir: Path,
    preview_memberships: int,
    overlapping_n_communities: int | None,
) -> None:
    """Run the overlapping extension on the DBLP co-authorship network."""
    print("\nLoading DBLP ...")
    from data.load_dblp import load_dblp
    G, ground_truth = load_dblp()

    print(f"\n{'='*65}")
    print("  DATASET: DBLP Co-authorship")
    print(f"  Nodes={G.number_of_nodes()}  Edges={G.number_of_edges()}")
    print(f"  Ground-truth communities: {len(ground_truth)}")
    print(f"{'='*65}")

    print("\n>> Running overlapping HP-MOCD-like extension ...")
    overlapping_partition, overlapping_rt = run_hp_mocd_overlapping(
        G,
        max_memberships=2,
        n_communities=overlapping_n_communities,
    )
    print(f"   Detected {len(overlapping_partition)} communities in {overlapping_rt:.2f}s")

    overlapping_scores = evaluate_overlapping(G, overlapping_partition, ground_truth)
    _print_scores("DBLP overlapping extension", overlapping_scores)

    overlapping_output = output_dir / "dblp_overlapping_extension_memberships.json"
    overlapping_node_map = save_partition_report(
        overlapping_output,
        "DBLP Co-authorship",
        "overlapping_extension",
        overlapping_partition,
    )
    _print_membership_preview("DBLP overlapping extension", overlapping_node_map, preview_memberships)
    print(f"   Saved overlapping memberships: {overlapping_output}")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HP-MOCD comparison experiment")
    parser.add_argument(
        "--dataset",
        choices=["lfr", "dblp", "both"],
        default="both",
        help="Which dataset to run (default: both)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Folder for baseline/overlapping membership reports (default: outputs)",
    )
    parser.add_argument(
        "--preview-memberships",
        type=int,
        default=25,
        help="How many node->community assignments to preview in console",
    )
    parser.add_argument(
        "--n-communities",
        type=int,
        default=None,
        help=(
            "Override overlapping initial label count. Useful when you observe "
            "artificially small community counts."
        ),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.dataset in ("lfr", "both"):
        run_lfr(
            output_dir=output_dir,
            preview_memberships=args.preview_memberships,
            overlapping_n_communities=args.n_communities,
        )

    if args.dataset in ("dblp", "both"):
        run_dblp(
            output_dir=output_dir,
            preview_memberships=args.preview_memberships,
            overlapping_n_communities=args.n_communities,
        )
