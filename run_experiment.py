"""
run_experiment.py
═════════════════
Runs all five methods on both datasets and prints a comparison table.

Methods compared
────────────────
  1. HP-MOCD baseline      — disjoint, evolutionary (pymocd / MinimalNSGAII)
  2. Overlapping extension — overlapping, evolutionary (your extension)
  3. MCMOEA                - overlapping, multi-community evolutionary
  4. CPM original          — overlapping, structural, NCN problem present
  5. CPM NCN-fixed         — overlapping, structural, all nodes covered

Usage
─────
    python run_experiment.py                  # both datasets
    python run_experiment.py --dataset lfr
    python run_experiment.py --dataset dblp
"""

import sys
import argparse
import json
import os
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import networkx as nx

from HPMOCD.hp_mocd_baseline        import run_hp_mocd
from HPMOCD.hp_mocd_overlapping     import run_hp_mocd_overlapping
from HPMOCD.mcmoea                  import run_mcmoea
from HPMOCD.cpm_community_detection import run_cpm_original, run_cpm_ncn_fixed
from evaluation.metrics               import evaluate_disjoint, evaluate_overlapping


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _count_overlapping_nodes(partition: list[frozenset]) -> int:
    node_counts: Counter = Counter()
    for community in partition:
        for node in community:
            node_counts[node] += 1
    return sum(1 for c in node_counts.values() if c > 1)


def _count_assigned_nodes(partition: list[frozenset]) -> int:
    return len(set().union(*partition)) if partition else 0


def _save_memberships(
    partition: list[frozenset],
    label: str,
    dataset_name: str,
    output_dir: str = "outputs",
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    node_to_communities: dict[str, list] = {}
    for cid, community in enumerate(partition):
        for node in community:
            node_to_communities.setdefault(str(node), []).append(cid)

    safe_dataset = dataset_name.lower().replace(" ", "_")
    safe_label   = label.lower().replace(" ", "_").replace("-", "_")
    path = os.path.join(output_dir, f"{safe_dataset}_{safe_label}_memberships.json")
    with open(path, "w") as f:
        json.dump(node_to_communities, f)
    print(f"   Saved memberships: {path}")


def _print_method_result(
    label: str,
    partition: list[frozenset],
    runtime: float,
    n_total_nodes: int,
    scores: dict[str, float],
    extra_info: str = "",
) -> None:
    n_assigned    = _count_assigned_nodes(partition)
    n_overlapping = _count_overlapping_nodes(partition)
    print(f"\n>> {label}")
    print(f"   Communities: {len(partition)} | "
          f"Runtime: {runtime:.2f}s | "
          f"Overlapping nodes: {n_overlapping}/{n_total_nodes} | "
          f"Assigned: {n_assigned}/{n_total_nodes}"
          + (f" | {extra_info}" if extra_info else ""))
    score_str = "  |  ".join(f"{k}={v:.4f}" for k, v in scores.items())
    print(f"  [{label}]  {score_str}")


def _print_comparison_table(results: list[dict], metrics: list[str]) -> None:
    col_w   = 16
    label_w = 22
    width   = label_w + col_w * len(results) + 4

    print(f"\n{'-' * width}")
    header = f"  {'Metric':<{label_w}}"
    for r in results:
        header += f"{r['label']:>{col_w}}"
    print(header)
    print(f"{'-' * width}")

    for metric in metrics:
        row    = f"  {metric:<{label_w}}"
        values = [r["scores"].get(metric, float("nan")) for r in results]
        # best = max, ignoring nan
        valid  = [v for v in values if v == v]
        best_val = max(valid) if valid else float("nan")
        for v in values:
            if v != v:  # nan
                cell = "N/A"
                row += f"{cell:>{col_w}}"
            else:
                marker = "<" if abs(v - best_val) < 1e-6 else " "
                row += f"{v:>{col_w - 2}.4f} {marker} "
        print(row)

    print(f"{'-' * width}")

    row = f"  {'Runtime (s)':<{label_w}}"
    for r in results:
        row += f"{r['runtime']:>{col_w - 1}.2f}s"
    print(row)

    row = f"  {'Communities':<{label_w}}"
    for r in results:
        row += f"{r['n_communities']:>{col_w}}"
    print(row)

    row = f"  {'Overlap nodes':<{label_w}}"
    for r in results:
        row += f"{r['n_overlapping']:>{col_w}}"
    print(row)

    row = f"  {'Assigned nodes':<{label_w}}"
    for r in results:
        row += f"{r['n_assigned']:>{col_w}}"
    print(row)

    print(f"{'-' * width}")
    print("  < = best value for that metric\n")


# ══════════════════════════════════════════════════════════════════════════════
#  CORE: run all methods on one graph
# ══════════════════════════════════════════════════════════════════════════════

def _run_all_methods(
    G: nx.Graph,
    ground_truth: list[frozenset],
    dataset_name: str,
    is_overlapping_gt: bool,
) -> None:
    n_nodes = G.number_of_nodes()

    print(f"\n{'=' * 65}")
    print(f"  DATASET : {dataset_name}")
    print(f"  Nodes   : {n_nodes}   Edges: {G.number_of_edges()}")
    print(f"  Ground-truth communities : {len(ground_truth)}")
    gt_overlapping = _count_overlapping_nodes(ground_truth)
    print(f"  Ground-truth overlapping : {gt_overlapping}/{n_nodes} "
          f"({100 * gt_overlapping / n_nodes:.1f}%)")
    print(f"{'=' * 65}")

    evaluate_fn = evaluate_overlapping if is_overlapping_gt else evaluate_disjoint
    results = []

    # ── 1. HP-MOCD baseline ───────────────────────────────────────────────────
    print("\n>> Running HP-MOCD baseline (disjoint) ...")
    p_base, _, rt_base = run_hp_mocd(G)
    s_base = evaluate_fn(G, p_base, ground_truth)
    _print_method_result("HP-MOCD Baseline", p_base, rt_base, n_nodes, s_base)
    _save_memberships(p_base, "baseline", dataset_name)
    results.append({
        "label": "HP-MOCD",
        "scores": s_base, "runtime": rt_base,
        "n_communities": len(p_base),
        "n_overlapping": _count_overlapping_nodes(p_base),
        "n_assigned":    _count_assigned_nodes(p_base),
    })

    # ── 2. Overlapping extension ──────────────────────────────────────────────
    print("\n>> Running overlapping HP-MOCD extension ...")
    p_ovlp, rt_ovlp = run_hp_mocd_overlapping(G, max_memberships=2)
    s_ovlp = evaluate_fn(G, p_ovlp, ground_truth)
    _print_method_result("HP-MOCD Overlapping", p_ovlp, rt_ovlp, n_nodes, s_ovlp)
    _save_memberships(p_ovlp, "overlapping", dataset_name)
    results.append({
        "label": "HP-MOCD-Ovlp",
        "scores": s_ovlp, "runtime": rt_ovlp,
        "n_communities": len(p_ovlp),
        "n_overlapping": _count_overlapping_nodes(p_ovlp),
        "n_assigned":    _count_assigned_nodes(p_ovlp),
    })

    # -- 3. MCMOEA ------------------------------------------------------------
    print("\n>> Running MCMOEA (multi-community multi-objective evolutionary) ...")
    p_mcmoea, rt_mcmoea = run_mcmoea(G, max_memberships=2)
    s_mcmoea = evaluate_fn(G, p_mcmoea, ground_truth)
    _print_method_result("MCMOEA", p_mcmoea, rt_mcmoea, n_nodes, s_mcmoea)
    _save_memberships(p_mcmoea, "mcmoea", dataset_name)
    results.append({
        "label": "MCMOEA",
        "scores": s_mcmoea, "runtime": rt_mcmoea,
        "n_communities": len(p_mcmoea),
        "n_overlapping": _count_overlapping_nodes(p_mcmoea),
        "n_assigned":    _count_assigned_nodes(p_mcmoea),
    })

    print("\n>> Running CPM original (NCN problem present) ...")
    p_cpm_orig, rt_cpm_orig, k_orig = run_cpm_original(G)
    n_unassigned = n_nodes - _count_assigned_nodes(p_cpm_orig)
    if p_cpm_orig:
        s_cpm_orig = evaluate_fn(G, p_cpm_orig, ground_truth)
    else:
        s_cpm_orig = {m: float("nan") for m in s_base.keys()}
    _print_method_result(
        f"CPM Original (k={k_orig})", p_cpm_orig, rt_cpm_orig, n_nodes, s_cpm_orig,
        extra_info=f"unassigned={n_unassigned}",
    )
    _save_memberships(p_cpm_orig, "cpm_original", dataset_name)
    results.append({
        "label": f"CPM-Orig(k={k_orig})",
        "scores": s_cpm_orig, "runtime": rt_cpm_orig,
        "n_communities": len(p_cpm_orig),
        "n_overlapping": _count_overlapping_nodes(p_cpm_orig),
        "n_assigned":    _count_assigned_nodes(p_cpm_orig),
    })

    # ── 4. CPM NCN-fixed ──────────────────────────────────────────────────────
    print("\n>> Running CPM NCN-fixed (all nodes assigned) ...")
    p_cpm_fix, rt_cpm_fix, k_fix = run_cpm_ncn_fixed(G)
    s_cpm_fix = evaluate_fn(G, p_cpm_fix, ground_truth)
    _print_method_result(
        f"CPM NCN-Fixed (k={k_fix})", p_cpm_fix, rt_cpm_fix, n_nodes, s_cpm_fix,
    )
    _save_memberships(p_cpm_fix, "cpm_ncn_fixed", dataset_name)
    results.append({
        "label": f"CPM-Fixed(k={k_fix})",
        "scores": s_cpm_fix, "runtime": rt_cpm_fix,
        "n_communities": len(p_cpm_fix),
        "n_overlapping": _count_overlapping_nodes(p_cpm_fix),
        "n_assigned":    _count_assigned_nodes(p_cpm_fix),
    })

    # ── Comparison table ──────────────────────────────────────────────────────
    _print_comparison_table(results, list(s_base.keys()))


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET RUNNERS
# ══════════════════════════════════════════════════════════════════════════════

def run_lfr() -> None:
    print("\nLoading LFR benchmark ...")
    try:
        from data.load_lfr import load_lfr_overlapping
        G, ground_truth = load_lfr_overlapping()
        label = "LFR Overlapping Benchmark"
        is_overlapping = True
    except TypeError as e:
        print(f"  [WARNING] Overlapping LFR not available ({e}), using disjoint.")
        from data.load_lfr import load_lfr_disjoint
        G, ground_truth = load_lfr_disjoint()
        label = "LFR Disjoint Benchmark"
        is_overlapping = False
    _run_all_methods(G, ground_truth, label, is_overlapping_gt=is_overlapping)


def run_dblp() -> None:
    print("\nLoading DBLP ...")
    from data.load_dblp import load_dblp
    G, ground_truth = load_dblp()
    _run_all_methods(G, ground_truth, "DBLP Co-authorship", is_overlapping_gt=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["lfr", "dblp", "both"], default="both")
    args = parser.parse_args()
    if args.dataset in ("lfr", "both"):
        run_lfr()
    if args.dataset in ("dblp", "both"):
        run_dblp()
