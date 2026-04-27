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

from HPMOCD.hp_mocd_baseline       import run_hp_mocd, HAS_HPMOCD
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


def _run_both_methods(
    G: nx.Graph,
    ground_truth: list,
    dataset_name: str,
    is_overlapping_gt: bool,
    output_dir: Path,
    preview_memberships: int,
    overlapping_n_communities: int | None,
) -> None:
    """
    Run baseline + overlapping on one graph and print comparison.

    Parameters
    ----------
    G                  : the graph
    ground_truth       : list[frozenset] ground-truth communities
    dataset_name       : string label for printing
    is_overlapping_gt  : True if ground truth has overlapping communities
                         (uses evaluate_overlapping which includes Omega)
    """
    print(f"\n{'='*65}")
    print(f"  DATASET: {dataset_name}")
    print(f"  Nodes={G.number_of_nodes()}  Edges={G.number_of_edges()}")
    print(f"  Ground-truth communities: {len(ground_truth)}")
    print(f"{'='*65}")

    if not HAS_HPMOCD:
        fallback_init = max(2, int(len(G) ** 0.5 // 3))
        print("  [DIAG] Baseline is using MinimalNSGAII fallback "
              "because pymocd is not installed.")
        print("  [DIAG] Fallback initial labels="
              f"{fallback_init}; this acts as an upper bound because "
              "fallback operators do not create new labels.")

    overlap_init = (
        max(2, overlapping_n_communities)
        if overlapping_n_communities is not None
        else max(2, int(len(G) ** 0.5 // 2))
    )
    print("  [DIAG] Overlapping initial labels="
            f"{overlap_init}; this is an initial label pool (not a hard cap), "
            "but very small values can still bias toward fewer communities.")

    evaluate_fn = evaluate_overlapping if is_overlapping_gt else evaluate_disjoint
    dataset_slug = _slugify(dataset_name)

    # ── 1. Baseline HP-MOCD ───────────────────────────────────────────────────
    print("\n>> Running baseline HP-MOCD ...")
    baseline_partition, _, baseline_rt = run_hp_mocd(G)
    print(f"   Detected {len(baseline_partition)} communities in {baseline_rt:.2f}s")

    baseline_scores = evaluate_fn(G, baseline_partition, ground_truth)
    _print_scores("Baseline", baseline_scores)

    baseline_output = output_dir / f"{dataset_slug}_baseline_memberships.json"
    baseline_node_map = save_partition_report(
        baseline_output,
        dataset_name,
        "baseline",
        baseline_partition,
    )
    _print_membership_preview("Baseline", baseline_node_map, preview_memberships)
    print(f"   Saved baseline memberships: {baseline_output}")

    # ── 2. Overlapping extension ──────────────────────────────────────────────
    print("\n>> Running overlapping HP-MOCD extension ...")
    overlapping_partition, overlapping_rt = run_hp_mocd_overlapping(
        G,
        max_memberships=2,
        n_communities=overlapping_n_communities,
    )
    print(f"   Detected {len(overlapping_partition)} communities in {overlapping_rt:.2f}s")

    # Count how many nodes actually overlap
    node_counts: dict[int, int] = {}
    for community in overlapping_partition:
        for node in community:
            node_counts[node] = node_counts.get(node, 0) + 1
    n_overlapping = sum(1 for c in node_counts.values() if c > 1)
    print(f"   Overlapping nodes: {n_overlapping} / {G.number_of_nodes()}")

    overlapping_scores = evaluate_fn(G, overlapping_partition, ground_truth)
    _print_scores("Overlapping", overlapping_scores)

    overlapping_output = output_dir / f"{dataset_slug}_overlapping_memberships.json"
    overlapping_node_map = save_partition_report(
        overlapping_output,
        dataset_name,
        "overlapping",
        overlapping_partition,
    )
    _print_membership_preview("Overlapping", overlapping_node_map, preview_memberships)
    print(f"   Saved overlapping memberships: {overlapping_output}")

    # ── Summary comparison ────────────────────────────────────────────────────
    print(f"\n{'-'*65}")
    print(f"  {'Metric':<15} {'Baseline':>12} {'Overlapping':>14} {'Better':>10}")
    print(f"{'-'*65}")
    for metric in baseline_scores:
        b = baseline_scores[metric]
        o = overlapping_scores[metric]
        better = "Baseline" if b >= o else "Overlapping"
        print(f"  {metric:<15} {b:>12.4f} {o:>14.4f} {better:>10}")
    print(f"{'-'*65}")
    print(f"  {'Runtime':<15} {baseline_rt:>11.2f}s {overlapping_rt:>13.2f}s")
    print(f"{'-'*65}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET LOADERS
# ══════════════════════════════════════════════════════════════════════════════

def run_lfr(
    output_dir: Path,
    preview_memberships: int,
    overlapping_n_communities: int | None,
) -> None:
    """Run both methods on the LFR overlapping benchmark."""
    print("\nLoading LFR benchmark ...")
    try:
        from data.load_lfr import load_lfr_overlapping
        G, ground_truth = load_lfr_overlapping()
    except TypeError as e:
        # Older NetworkX versions don't support the 'on'/'om' overlap params.
        # Fall back to the disjoint LFR variant.
        print(f"  [WARNING] Overlapping LFR failed ({e})")
        print("  Falling back to disjoint LFR ...")
        from data.load_lfr import load_lfr_disjoint
        G, ground_truth = load_lfr_disjoint()
        _run_both_methods(
            G,
            ground_truth,
            "LFR (disjoint fallback)",
            is_overlapping_gt=False,
            output_dir=output_dir,
            preview_memberships=preview_memberships,
            overlapping_n_communities=overlapping_n_communities,
        )
        return

    _run_both_methods(
        G,
        ground_truth,
        "LFR Overlapping Benchmark",
        is_overlapping_gt=True,
        output_dir=output_dir,
        preview_memberships=preview_memberships,
        overlapping_n_communities=overlapping_n_communities,
    )


def run_dblp(
    output_dir: Path,
    preview_memberships: int,
    overlapping_n_communities: int | None,
) -> None:
    """Run both methods on the DBLP co-authorship network."""
    print("\nLoading DBLP ...")
    from data.load_dblp import load_dblp
    G, ground_truth = load_dblp()
    _run_both_methods(
        G,
        ground_truth,
        "DBLP Co-authorship",
        is_overlapping_gt=True,
        output_dir=output_dir,
        preview_memberships=preview_memberships,
        overlapping_n_communities=overlapping_n_communities,
    )


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

    # if args.dataset in ("lfr", "both"):
    #     run_lfr(
    #         output_dir=output_dir,
    #         preview_memberships=args.preview_memberships,
    #         overlapping_n_communities=args.n_communities,
    #     )

    if args.dataset in ("dblp", "both"):
        run_dblp(
            output_dir=output_dir,
            preview_memberships=args.preview_memberships,
            overlapping_n_communities=args.n_communities,
        )
