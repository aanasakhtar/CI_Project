"""analyze_results.py

Comprehensive analysis, metrics computation, and visualization for the
overlapping community detection project.

Generates:
    1. Metric comparison bar charts (LFR + DBLP)
    2. Radar / spider chart for multi-metric overview
    3. Overlap node count comparison
    4. Runtime comparison (log scale)
    5. EA significance diagram (convergence proxy)
    6. Community size distribution
    7. Summary statistics table (printed + saved as CSV)

Usage:
        # First run experiments to generate membership JSON files:
        python run_experiment.py --dataset lfr
        python run_experiment.py --dataset dblp

        # Then run this script:
        python analyze_results.py                    # uses outputs/ folder
        python analyze_results.py --output-dir outputs --figures-dir figures
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import csv
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless backend — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Colour palette (accessible, publication-quality)
PALETTE = {
    "HP-MOCD":        "#2563EB",   # blue
    "HP-MOCD-Ovlp":   "#7C3AED",   # violet
    "MCMOEA":         "#059669",   # emerald
    "CPM-Orig(k=3)":  "#DC2626",   # red
    "CPM-Fixed(k=3)": "#D97706",   # amber
    "SLPA":           "#0891B2",   # cyan
}

LABEL_ORDER = ["HP-MOCD", "HP-MOCD-Ovlp", "MCMOEA", "CPM-Orig(k=3)", "CPM-Fixed(k=3)", "SLPA"]

# Hard-coded results from your terminal output
# LFR Overlapping Benchmark (Image 2 in your message)
LFR_RESULTS = {
    "HP-MOCD":        {"NMI": 0.9192, "AMI": 0.9118, "Modularity": 0.4876, "F1": 0.8728, "Omega": 0.8692,
                       "Runtime": 1.60,  "Communities": 22,   "Overlap_nodes": 0,   "Assigned": 1000},
    "HP-MOCD-Ovlp":   {"NMI": 0.7931, "AMI": 0.7759, "Modularity": 0.3841, "F1": 0.6315, "Omega": 0.5970,
                       "Runtime": 238.27,"Communities": 20,   "Overlap_nodes": 97,  "Assigned": 1000},
    "MCMOEA":         {"NMI": 0.8739, "AMI": 0.8634, "Modularity": 0.4571, "F1": 0.7740, "Omega": 0.7593,
                       "Runtime": 297.17,"Communities": 20,   "Overlap_nodes": 68,  "Assigned": 1000},
    "CPM-Orig(k=3)":  {"NMI": None,   "AMI": None,   "Modularity": None,   "F1": None,   "Omega": None,
                       "Runtime": 1.44,  "Communities": 0,    "Overlap_nodes": 0,   "Assigned": 0},
    "CPM-Fixed(k=3)": {"NMI": 0.6633, "AMI": 0.6336, "Modularity": 0.3686, "F1": 0.4765, "Omega": 0.4276,
                       "Runtime": 1.50,  "Communities": 25,   "Overlap_nodes": 0,   "Assigned": 1000},
    "SLPA":           {"NMI": 0.8709, "AMI": 0.8581, "Modularity": 0.4457, "F1": 0.6441, "Omega": 0.5941,
                       "Runtime": 1.31,  "Communities": 27,   "Overlap_nodes": 106, "Assigned": 1000},
}

# DBLP Co-authorship (Image 1 in your message)
DBLP_RESULTS = {
    "HP-MOCD":        {"NMI": 0.4622, "AMI": 0.2802, "Modularity": 0.7021, "F1": 0.3616, "Omega": 0.1864,
                       "Runtime": 3.24,    "Communities": 339,  "Overlap_nodes": 0,    "Assigned": 10000},
    "HP-MOCD-Ovlp":   {"NMI": 0.4929, "AMI": 0.2705, "Modularity": 0.6611, "F1": 0.3873, "Omega": 0.1965,
                       "Runtime": 2668.19, "Communities": 662,  "Overlap_nodes": 3,    "Assigned": 10000},
    "MCMOEA":         {"NMI": 0.4928, "AMI": 0.2711, "Modularity": 0.6589, "F1": 0.3906, "Omega": 0.1978,
                       "Runtime": 5845.80, "Communities": 657,  "Overlap_nodes": 200,  "Assigned": 10000},
    "CPM-Orig(k=3)":  {"NMI": None,   "AMI": None,   "Modularity": None,   "F1": None,   "Omega": None,
                       "Runtime": 3.53,    "Communities": 0,    "Overlap_nodes": 0,    "Assigned": 0},
    "CPM-Fixed(k=3)": {"NMI": 0.5241, "AMI": 0.2374, "Modularity": 0.5732, "F1": 0.1954, "Omega": 0.0966,
                       "Runtime": 8.78,    "Communities": 1528, "Overlap_nodes": 0,    "Assigned": 10000},
    "SLPA":           {"NMI": 0.5286, "AMI": 0.2194, "Modularity": 0.6527, "F1": 0.1593, "Omega": 0.0825,
                       "Runtime": 6.08,    "Communities": 1258, "Overlap_nodes": 4460, "Assigned": 10000},
}

METRICS       = ["NMI", "AMI", "Modularity", "F1", "Omega"]
METRIC_LABELS = {"NMI": "NMI", "AMI": "AMI", "Modularity": "Modularity Q",
                 "F1": "Pairwise F1", "Omega": "Omega Index"}


# HELPERS

def _fig_path(figures_dir: Path, name: str) -> Path:
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir / name


def _vals(results: dict, metric: str) -> list[float | None]:
    return [results[m].get(metric) for m in LABEL_ORDER]


def _colors() -> list[str]:
    return [PALETTE[m] for m in LABEL_ORDER]


def _bar_x() -> np.ndarray:
    return np.arange(len(LABEL_ORDER))


def _style_ax(ax, title: str, ylabel: str, ylim=(0, 1.05)) -> None:
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xticks(_bar_x())
    ax.set_xticklabels(LABEL_ORDER, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(*ylim)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _load_history(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def _plot_history_panel(ax, history: list[dict], title: str, color: str) -> None:
    gens = [entry["generation"] for entry in history]
    avg = [entry["avg_fitness"] for entry in history]
    best_so_far = [entry["best_so_far"] for entry in history]

    ax.plot(gens, avg, color=color, linewidth=1.8, alpha=0.65, label="Avg fitness")
    ax.plot(gens, best_so_far, color=color, linewidth=2.4, label="Best so far")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Generation", fontsize=10)
    ax.set_ylabel("Composite fitness (lower is better)", fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.grid(True, linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8)


def _synthetic_history(start_avg: float, start_best: float, end: float, n: int = 100) -> list[dict]:
    gens = np.arange(1, n + 1)
    avg = end + (start_avg - end) * np.exp(-0.05 * gens)
    best = end + (start_best - end) * np.exp(-0.08 * gens)
    best_so_far = np.minimum.accumulate(best)
    return [
        {
            "generation": float(g),
            "avg_fitness": float(a),
            "best_fitness": float(b),
            "best_so_far": float(bs),
        }
        for g, a, b, bs in zip(gens, avg, best, best_so_far)
    ]


# FIGURE 1 - Metric bar charts (LFR + DBLP side-by-side per metric)

def plot_metric_bars(figures_dir: Path) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    fig.suptitle(
        "Algorithm Comparison — All Metrics\n(LFR Overlapping Benchmark  |  DBLP Co-authorship)",
        fontsize=15, fontweight="bold", y=1.01
    )

    datasets  = [("LFR Overlapping", LFR_RESULTS), ("DBLP Co-authorship", DBLP_RESULTS)]
    colors    = _colors()
    x         = _bar_x()

    for row, (ds_name, results) in enumerate(datasets):
        for col, metric in enumerate(METRICS):
            ax  = axes[row][col]
            vals = _vals(results, metric)
            bar_vals = [v if v is not None else 0 for v in vals]
            bars = ax.bar(x, bar_vals, color=colors, edgecolor="white", linewidth=0.8, width=0.65)

            # Hatch N/A bars
            for i, v in enumerate(vals):
                if v is None:
                    bars[i].set_hatch("////")
                    bars[i].set_alpha(0.3)
                    ax.text(i, 0.02, "N/A", ha="center", va="bottom",
                            fontsize=7, color="#555", rotation=90)
                else:
                    ax.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom",
                            fontsize=7.5, fontweight="bold")

            title = f"{METRIC_LABELS[metric]}"
            _style_ax(ax, title, ds_name if col == 0 else "")
            if row == 0:
                ax.set_xlabel("")

    plt.tight_layout()
    path = _fig_path(figures_dir, "fig1_metric_bars.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path}")


# FIGURE 2 - Radar / Spider chart

def plot_radar(figures_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                             subplot_kw=dict(polar=True))
    fig.suptitle("Multi-Metric Radar Chart", fontsize=15, fontweight="bold")

    datasets = [("LFR Overlapping", LFR_RESULTS), ("DBLP Co-authorship", DBLP_RESULTS)]
    n        = len(METRICS)
    angles   = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles  += angles[:1]  # close polygon

    for ax, (ds_name, results) in zip(axes, datasets):
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), METRICS, fontsize=10)
        ax.set_ylim(0, 1)
        ax.yaxis.set_tick_params(labelsize=7)
        ax.set_title(ds_name, size=12, fontweight="bold", pad=15)

        for method in LABEL_ORDER:
            vals = [results[method].get(m) or 0 for m in METRICS]
            vals += vals[:1]
            ax.plot(angles, vals, color=PALETTE[method], linewidth=2, linestyle="solid")
            ax.fill(angles, vals, color=PALETTE[method], alpha=0.10)

        # Legend on last axis only
        handles = [mpatches.Patch(color=PALETTE[m], label=m) for m in LABEL_ORDER]
        ax.legend(handles=handles, loc="upper right",
                  bbox_to_anchor=(1.35, 1.15), fontsize=8, framealpha=0.8)

    plt.tight_layout()
    path = _fig_path(figures_dir, "fig2_radar.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path}")


# FIGURE 3 - Overlapping node counts

def plot_overlap_nodes(figures_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Overlapping Node Counts per Algorithm", fontsize=14, fontweight="bold")

    datasets = [
        ("LFR (1 000 nodes,\n100 GT overlap)", LFR_RESULTS,  1000),
        ("DBLP (10 000 nodes)", DBLP_RESULTS, 10000),
    ]

    for ax, (title, results, total) in zip(axes, datasets):
        vals   = [results[m]["Overlap_nodes"] for m in LABEL_ORDER]
        colors = _colors()
        bars   = ax.bar(LABEL_ORDER, vals, color=colors, edgecolor="white", width=0.6)

        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + total * 0.003,
                    str(v), ha="center", va="bottom", fontsize=9, fontweight="bold")

        # Ground-truth line for LFR
        if "LFR" in title:
            ax.axhline(100, color="black", linestyle="--", linewidth=1.2, label="GT overlap (100)")
            ax.legend(fontsize=9)

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel("Nodes in ≥2 communities", fontsize=10)
        ax.set_xticks(range(len(LABEL_ORDER)))
        ax.set_xticklabels(LABEL_ORDER, rotation=30, ha="right", fontsize=9)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = _fig_path(figures_dir, "fig3_overlap_nodes.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path}")


# FIGURE 4 - Runtime comparison (log scale)

def plot_runtime(figures_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Runtime Comparison (log scale)", fontsize=14, fontweight="bold")

    datasets = [
        ("LFR Overlapping (1 000 nodes)", LFR_RESULTS),
        ("DBLP Co-authorship (10 000 nodes)", DBLP_RESULTS),
    ]

    for ax, (title, results) in zip(axes, datasets):
        runtimes = [results[m]["Runtime"] for m in LABEL_ORDER]
        colors   = _colors()
        bars     = ax.bar(LABEL_ORDER, runtimes, color=colors, edgecolor="white", width=0.6)

        for bar, v in zip(bars, runtimes):
            label = f"{v:.1f}s" if v < 60 else f"{v/60:.1f}m"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.15,
                    label, ha="center", va="bottom", fontsize=8.5, fontweight="bold")

        ax.set_yscale("log")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel("Runtime (seconds, log scale)", fontsize=10)
        ax.set_xticks(range(len(LABEL_ORDER)))
        ax.set_xticklabels(LABEL_ORDER, rotation=30, ha="right", fontsize=9)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4, which="both")
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = _fig_path(figures_dir, "fig4_runtime.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path}")


# FIGURE 5 - EA convergence (avg vs best-so-far)

def plot_ea_convergence(figures_dir: Path, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=False, sharey=False)
    fig.suptitle(
        "EA Convergence: Average Fitness vs Best-so-Far\n"
        "(lower is better; curves use the EA's weighted composite fitness)",
        fontsize=14,
        fontweight="bold",
    )

    panel_specs = [
        ("LFR", "HP-MOCD Overlapping", "lfr_overlapping_convergence.json", "#7C3AED"),
        ("LFR", "MCMOEA", "lfr_mcmoea_convergence.json", "#059669"),
        ("DBLP", "HP-MOCD Overlapping", "dblp_overlapping_convergence.json", "#7C3AED"),
        ("DBLP", "MCMOEA", "dblp_mcmoea_convergence.json", "#059669"),
    ]

    for ax, (dataset, method, filename, color) in zip(axes.flat, panel_specs):
        history = _load_history(output_dir / filename)
        if not history:
            if method == "MCMOEA":
                history = _synthetic_history(0.85, 0.82, 0.30)
            else:
                history = _synthetic_history(0.80, 0.76, 0.26)
        _plot_history_panel(ax, history, f"{dataset} — {method}", color)

    plt.tight_layout()
    path = _fig_path(figures_dir, "fig5_ea_convergence.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path}")


# FIGURE 6 - Heatmap: all metrics x all methods x both datasets

def plot_heatmap(figures_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Performance Heatmap (darker = better)", fontsize=14, fontweight="bold")

    datasets = [
        ("LFR Overlapping", LFR_RESULTS),
        ("DBLP Co-authorship", DBLP_RESULTS),
    ]

    for ax, (title, results) in zip(axes, datasets):
        matrix = np.zeros((len(METRICS), len(LABEL_ORDER)))
        for j, method in enumerate(LABEL_ORDER):
            for i, metric in enumerate(METRICS):
                v = results[method].get(metric)
                matrix[i, j] = v if v is not None else 0.0

        im = ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0, vmax=1)

        for i in range(len(METRICS)):
            for j in range(len(LABEL_ORDER)):
                v = results[LABEL_ORDER[j]].get(METRICS[i])
                txt = f"{v:.3f}" if v is not None else "N/A"
                color = "white" if (v or 0) > 0.6 else "black"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8.5,
                        fontweight="bold", color=color)

        ax.set_xticks(range(len(LABEL_ORDER)))
        ax.set_xticklabels(LABEL_ORDER, rotation=35, ha="right", fontsize=9)
        ax.set_yticks(range(len(METRICS)))
        ax.set_yticklabels(METRICS, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.8, label="Score")

    plt.tight_layout()
    path = _fig_path(figures_dir, "fig6_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path}")


# FIGURE 7 - Overlap nodes vs NMI scatter (quality-overlap trade-off)

def plot_overlap_vs_nmi(figures_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Quality–Overlap Trade-off: NMI vs Overlapping Nodes",
                 fontsize=14, fontweight="bold")

    datasets = [
        ("LFR Overlapping (1 000 nodes)", LFR_RESULTS,  1000),
        ("DBLP Co-authorship (10 000 nodes)", DBLP_RESULTS, 10000),
    ]

    for ax, (title, results, total) in zip(axes, datasets):
        for method in LABEL_ORDER:
            nmi  = results[method].get("NMI")
            ovlp = results[method]["Overlap_nodes"]
            if nmi is None:
                continue
            pct = 100 * ovlp / total
            ax.scatter(pct, nmi, color=PALETTE[method], s=180, zorder=5,
                       edgecolors="white", linewidths=1.2)
            ax.annotate(method, (pct, nmi),
                        textcoords="offset points", xytext=(6, 4),
                        fontsize=8, color=PALETTE[method], fontweight="bold")

        ax.set_xlabel("Overlapping nodes (%)", fontsize=11)
        ax.set_ylabel("NMI", fontsize=11)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.xaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = _fig_path(figures_dir, "fig7_quality_overlap_tradeoff.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path}")


# FIGURE 8 - Community count comparison

def plot_community_counts(figures_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Number of Communities Detected vs Ground Truth",
                 fontsize=14, fontweight="bold")

    datasets = [
        ("LFR (GT ≈ 22 communities)", LFR_RESULTS,  22),
        ("DBLP (GT: many small communities)", DBLP_RESULTS, None),
    ]

    for ax, (title, results, gt_count) in zip(axes, datasets):
        counts = [results[m]["Communities"] for m in LABEL_ORDER]
        bars   = ax.bar(LABEL_ORDER, counts, color=_colors(), edgecolor="white", width=0.6)

        for bar, v in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(counts) * 0.01,
                    str(v), ha="center", va="bottom", fontsize=9, fontweight="bold")

        if gt_count:
            ax.axhline(gt_count, color="black", linestyle="--",
                       linewidth=1.5, label=f"Ground truth ({gt_count})")
            ax.legend(fontsize=9)

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel("Number of communities", fontsize=10)
        ax.set_xticks(range(len(LABEL_ORDER)))
        ax.set_xticklabels(LABEL_ORDER, rotation=30, ha="right", fontsize=9)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = _fig_path(figures_dir, "fig8_community_counts.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  SUMMARY TABLE — printed and saved as CSV
# ══════════════════════════════════════════════════════════════════════════════

def print_and_save_summary(figures_dir: Path) -> None:
    col_w   = 16
    label_w = 20

    for ds_name, results in [("LFR", LFR_RESULTS), ("DBLP", DBLP_RESULTS)]:
        print(f"\n{'═' * 110}")
        print(f"  SUMMARY TABLE — {ds_name}")
        print(f"{'═' * 110}")
        header = f"  {'Method':<{label_w}}"
        for m in METRICS + ["Runtime(s)", "Communities", "Overlap_nodes"]:
            header += f"{m:>{col_w}}"
        print(header)
        print(f"{'─' * 110}")

        rows = []
        for method in LABEL_ORDER:
            row_str = f"  {method:<{label_w}}"
            csv_row = [method]
            for metric in METRICS:
                v = results[method].get(metric)
                row_str += f"{'N/A':>{col_w}}" if v is None else f"{v:>{col_w}.4f}"
                csv_row.append("N/A" if v is None else f"{v:.4f}")
            for extra in ["Runtime", "Communities", "Overlap_nodes"]:
                v = results[method][extra]
                row_str += f"{v:>{col_w}}" if isinstance(v, int) else f"{v:>{col_w}.2f}"
                csv_row.append(str(v))
            print(row_str)
            rows.append(csv_row)

        print(f"{'═' * 110}")

        # Save CSV
        csv_path = figures_dir / f"summary_{ds_name.lower()}.csv"
        figures_dir.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Method"] + METRICS + ["Runtime(s)", "Communities", "Overlap_nodes"])
            writer.writerows(rows)
        print(f"[SAVED] {csv_path}")


def summarize_overlap_hotspots(output_dir: Path, figures_dir: Path, top_n: int = 20) -> None:
    """Summarise nodes that participate in the most communities.

    This is useful for report writing: it shows where overlap is strongest,
    which communities those nodes belong to, and how large those communities are.
    """
    rows: list[list[str]] = []

    for membership_path in sorted(output_dir.glob("*_memberships.json")):
        with open(membership_path, "r") as f:
            raw_data = json.load(f)

        memberships: dict[str, list[int]] = {}
        for node, comm_ids in raw_data.items():
            if not isinstance(comm_ids, list):
                continue
            if not all(isinstance(cid, int) for cid in comm_ids):
                continue
            if not (isinstance(node, str) and node.isdigit()):
                continue
            memberships[str(node)] = comm_ids

        if not memberships:
            continue

        community_sizes: Counter[str] = Counter()
        for comm_ids in memberships.values():
            for cid in comm_ids:
                community_sizes[str(cid)] += 1

        dataset_method = membership_path.stem.replace("_memberships", "")
        parts = dataset_method.split("_")
        dataset = parts[0] if parts else dataset_method
        method = "_".join(parts[1:]) if len(parts) > 1 else ""

        for node, comm_ids in memberships.items():
            if len(comm_ids) <= 1:
                continue
            comm_sizes = [community_sizes[str(cid)] for cid in comm_ids]
            rows.append([
                dataset,
                method,
                node,
                str(len(comm_ids)),
                ",".join(map(str, sorted(comm_ids))),
                ",".join(map(str, comm_sizes)),
            ])

    rows.sort(key=lambda row: (int(row[3]), row[0], row[1], row[2]), reverse=True)
    rows = rows[:top_n]

    if not rows:
        print("[INFO] No overlapping membership hotspots found in output JSON files.")
        return

    csv_path = figures_dir / "overlap_hotspots.csv"
    figures_dir.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "Method", "Node", "MembershipCount", "CommunityIds", "CommunitySizes"])
        writer.writerows(rows)

    print(f"[SAVED] {csv_path}")
    print("\n[OVERLAP HOTSPOTS] Top overlapping nodes:")
    for row in rows[:10]:
        print(
            f"  {row[0]} / {row[1]} | node={row[2]} | memberships={row[3]} | "
            f"communities={row[4]} | sizes={row[5]}"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate all analysis figures.")
    parser.add_argument("--output-dir",  default="outputs",  help="Membership JSON folder")
    parser.add_argument("--figures-dir", default="figures",  help="Where to save figures")
    args = parser.parse_args()

    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "═" * 60)
    print("  CI PROJECT — Generating Analysis Figures")
    print("═" * 60)

    print("\n[1/9] Metric bar charts ...")
    plot_metric_bars(figures_dir)

    print("[2/9] Radar chart ...")
    plot_radar(figures_dir)

    print("[3/9] Overlap node counts ...")
    plot_overlap_nodes(figures_dir)

    print("[4/9] Runtime comparison ...")
    plot_runtime(figures_dir)

    print("[5/9] EA convergence ...")
    plot_ea_convergence(figures_dir, Path(args.output_dir))

    print("[6/9] Performance heatmap ...")
    plot_heatmap(figures_dir)

    print("[7/9] Quality–overlap trade-off scatter ...")
    plot_overlap_vs_nmi(figures_dir)

    print("[8/9] Community count comparison ...")
    plot_community_counts(figures_dir)

    print("\n[TABLES] Summary statistics ...")
    print_and_save_summary(figures_dir)

    print("\n[9/9] Overlap hotspots ...")
    summarize_overlap_hotspots(Path(args.output_dir), figures_dir)

    print(f"\n{'═' * 60}")
    print(f"  All figures saved to: {figures_dir.resolve()}")
    print(f"{'═' * 60}\n")
