
import sys
import argparse
import json
import os
import traceback
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

print(f"[INFO] Python {sys.version}")
print(f"[INFO] Project root: {PROJECT_ROOT}")

# ── Verify files exist before importing ───────────────────────────────────
slpa_path = PROJECT_ROOT / "HPMOCD" / "slpa.py"
if not slpa_path.exists():
    print(f"\n[ERROR] slpa.py not found at: {slpa_path}")
    print("  Copy slpa.py into your HPMOCD/ folder and re-run.")
    sys.exit(1)

print(f"[OK] slpa.py found at              {slpa_path}")

# ── Imports ───────────────────────────────────────────────────────────────────
import networkx as nx

try:
    from HPMOCD.hp_mocd_baseline        import run_hp_mocd
    from HPMOCD.hp_mocd_overlapping     import run_hp_mocd_overlapping
    from HPMOCD.mcmoea                  import run_mcmoea
    from HPMOCD.cpm_community_detection import run_cpm_original, run_cpm_ncn_fixed
    from HPMOCD.slpa                    import run_slpa
    from evaluation.metrics             import evaluate_disjoint, evaluate_overlapping
    print("[OK] All imports successful.\n")
except ImportError as e:
    print(f"\n[IMPORT ERROR] {e}")
    traceback.print_exc()
    sys.exit(1)

ALL_METHODS = ["hpmocd", "hpmocd_ovlp", "mcmoea", "cpm_orig", "cpm_fixed", "slpa"]

# Evolution parameters
POP_SIZE = 100
GENERATIONS = 100


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _count_overlapping_nodes(partition: list) -> int:
    node_counts: Counter = Counter()
    for community in partition:
        for node in community:
            node_counts[node] += 1
    return sum(1 for c in node_counts.values() if c > 1)


def _count_assigned_nodes(partition: list) -> int:
    return len(set().union(*partition)) if partition else 0


def _save_memberships(
    partition: list,
    label: str,
    dataset_name: str,
    output_dir: str = "outputs",
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    node_to_communities: dict = {}
    for cid, community in enumerate(partition):
        for node in community:
            node_to_communities.setdefault(str(node), []).append(cid)

    safe_dataset = dataset_name.lower().replace(" ", "_")
    safe_label   = label.lower().replace(" ", "_").replace("-", "_")
    path = os.path.join(output_dir, f"{safe_dataset}_{safe_label}_memberships.json")
    with open(path, "w") as f:
        json.dump(node_to_communities, f)
    print(f"   Saved memberships → {path}")


def _print_method_result(
    label: str,
    partition: list,
    runtime: float,
    n_total_nodes: int,
    scores: dict,
    extra_info: str = "",
) -> None:
    n_assigned    = _count_assigned_nodes(partition)
    n_overlapping = _count_overlapping_nodes(partition)
    print(f"\n>> {label}")
    print(f"   Communities   : {len(partition)}")
    print(f"   Runtime       : {runtime:.2f}s")
    print(f"   Overlap nodes : {n_overlapping}/{n_total_nodes}")
    print(f"   Assigned nodes: {n_assigned}/{n_total_nodes}"
          + (f"  [{extra_info}]" if extra_info else ""))
    score_str = "  |  ".join(f"{k}={v:.4f}" for k, v in scores.items())
    print(f"   Scores → {score_str}")


def _print_comparison_table(results: list, metrics: list) -> None:
    col_w   = 18
    label_w = 16
    width   = label_w + col_w * len(results) + 4

    print(f"\n{'═' * width}")
    print("  COMPARISON TABLE")
    print(f"{'═' * width}")
    header = f"  {'Metric':<{label_w}}"
    for r in results:
        header += f"{r['label']:>{col_w}}"
    print(header)
    print(f"{'─' * width}")

    for metric in metrics:
        row    = f"  {metric:<{label_w}}"
        values = [r["scores"].get(metric, float("nan")) for r in results]
        valid  = [v for v in values if v == v]
        best_val = max(valid) if valid else float("nan")
        for v in values:
            if v != v:
                row += f"{'N/A':>{col_w}}"
            else:
                marker = "<" if abs(v - best_val) < 1e-6 else " "
                row += f"{v:>{col_w - 2}.4f} {marker} "
        print(row)

    print(f"{'─' * width}")
    
    # Runtime row
    row = f"  {'Runtime (s)':<{label_w}}"
    for r in results:
        row += f"{r['runtime']:>{col_w - 1}.2f}s"
    print(row)
    
    # Communities row
    row = f"  {'Communities':<{label_w}}"
    for r in results:
        row += f"{r['n_communities']:>{col_w}}"
    print(row)
    
    # Overlap nodes row
    row = f"  {'Overlap nodes':<{label_w}}"
    for r in results:
        row += f"{r['n_overlapping']:>{col_w}}"
    print(row)
    
    # Assigned nodes row
    row = f"  {'Assigned nodes':<{label_w}}"
    for r in results:
        row += f"{r['n_assigned']:>{col_w}}"
    print(row)

    print(f"{'═' * width}")
    print("  < = best value for that metric\n")


def _run_all_methods(
    G: nx.Graph,
    ground_truth: list,
    dataset_name: str,
    is_overlapping_gt: bool,
    methods: list,
) -> None:
    n_nodes = G.number_of_nodes()

    print(f"\n{'═' * 65}")
    print(f"  DATASET : {dataset_name}")
    print(f"  Nodes   : {n_nodes}   Edges: {G.number_of_edges()}")
    print(f"  Ground-truth communities : {len(ground_truth)}")
    gt_overlapping = _count_overlapping_nodes(ground_truth)
    print(f"  GT overlapping nodes     : {gt_overlapping}/{n_nodes} "
          f"({100 * gt_overlapping / n_nodes:.1f}%)")
    print(f"{'═' * 65}")

    evaluate_fn = evaluate_overlapping if is_overlapping_gt else evaluate_disjoint
    results = []

    def _safe_run(method_key, runner_fn, label, save_label):
        """Run a method with full error reporting."""
        print(f"\n{'─' * 50}")
        print(f"  Starting: {label}")
        print(f"{'─' * 50}")
        try:
            ret = runner_fn()
            partition, runtime = ret[0], ret[1]
            
            # skip evaluation if partition is empty
            if not partition or len(partition) == 0:
                print(f"  [WARNING] {label} returned empty partition, skipping evaluation")
                scores = {m: float("nan") for m in ["NMI", "AMI", "Modularity", "F1", "Omega"]}
            else:
                scores = evaluate_fn(G, partition, ground_truth)
                
            _print_method_result(label, partition, runtime, n_nodes, scores)
            _save_memberships(partition, save_label, dataset_name)
            results.append({
                "label": label,
                "scores": scores,
                "runtime": runtime,
                "n_communities": len(partition),
                "n_overlapping": _count_overlapping_nodes(partition),
                "n_assigned":    _count_assigned_nodes(partition),
            })
        except Exception as exc:
            print(f"  [ERROR in {label}]: {exc}")
            traceback.print_exc()

    # ── 1. HP-MOCD baseline (disjoint) ──────────────────────────────────────
    if "hpmocd" in methods:
        def _f1():
            p, _, rt = run_hp_mocd(G)  # No pop_size or n_gen needed
            return p, rt
        _safe_run("hpmocd", _f1, "HP-MOCD", "baseline")

    # ── 2. Overlapping extension ─────────────────────────────────────────────
    if "hpmocd_ovlp" in methods:
        def _f2():
            return run_hp_mocd_overlapping(G, max_memberships=2)  # No pop_size or n_gen
        _safe_run("hpmocd_ovlp", _f2, "HP-MOCD-Ovlp", "overlapping")

    # ── 3. MCMOEA ────────────────────────────────────────────────────────────
    if "mcmoea" in methods:
        def _f3():
            return run_mcmoea(G, max_memberships=2)  # No pop_size or n_gen
        _safe_run("mcmoea", _f3, "MCMOEA", "mcmoea")

    # ── 4. CPM Original ──────────────────────────────────────────────────────
    if "cpm_orig" in methods:
        def _f4():
            p, rt, _ = run_cpm_original(G)
            return p, rt
        _safe_run("cpm_orig", _f4, "CPM-Orig(k=3)", "cpm_original")

    # ── 5. CPM Fixed ─────────────────────────────────────────────────────────
    if "cpm_fixed" in methods:
        def _f5():
            p, rt, _ = run_cpm_ncn_fixed(G)
            return p, rt
        _safe_run("cpm_fixed", _f5, "CPM-Fixed(k=3)", "cpm_ncn_fixed")

    # ── 6. SLPA ──────────────────────────────────────────────────────────────
    if "slpa" in methods:
        def _f6():
            return run_slpa(G, T=20, r=0.1)
        _safe_run("slpa", _f6, "SLPA", "slpa")
    
    # ── Print comparison table ───────────────────────────────────────────────
    _print_comparison_table(results, ["NMI", "AMI", "Modularity", "F1", "Omega"])


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET RUNNERS
# ══════════════════════════════════════════════════════════════════════════════

def run_lfr(methods: list) -> None:
    print("\n[STEP] Loading LFR benchmark ...")
    try:
        from data.load_lfr import load_lfr_overlapping
        G, ground_truth = load_lfr_overlapping()
        label = "LFR Overlapping Benchmark"
        is_overlapping = True
        print(f"[OK] Loaded LFR overlapping: {G.number_of_nodes()} nodes")
    except Exception as e:
        print(f"[WARNING] Overlapping LFR failed ({e}), falling back to disjoint.")
        from data.load_lfr import load_lfr_disjoint
        G, ground_truth = load_lfr_disjoint()
        label = "LFR Disjoint Benchmark"
        is_overlapping = False
        print(f"[OK] Loaded LFR disjoint: {G.number_of_nodes()} nodes")
    _run_all_methods(G, ground_truth, label, is_overlapping, methods)


def run_dblp(methods: list) -> None:
    print("\n[STEP] Loading DBLP ...")
    from data.load_dblp import load_dblp
    G, ground_truth = load_dblp()
    print(f"[OK] Loaded DBLP: {G.number_of_nodes()} nodes")
    _run_all_methods(G, ground_truth, "DBLP Co-authorship", True, methods)


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run community detection methods and compare results."
    )
    parser.add_argument(
        "--dataset",
        choices=["lfr", "dblp", "both"],
        default="both",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=ALL_METHODS,
        default=ALL_METHODS,
        help=f"Methods to run. Options: {ALL_METHODS}",
    )
    args = parser.parse_args()

    print(f"[INFO] Dataset  : {args.dataset}")
    print(f"[INFO] Methods  : {args.methods}")
    print(f"[INFO] Evolution: pop_size={POP_SIZE}, generations={GENERATIONS}")

    if args.dataset in ("lfr", "both"):
        run_lfr(methods=args.methods)

    if args.dataset in ("dblp", "both"):
        run_dblp(methods=args.methods)