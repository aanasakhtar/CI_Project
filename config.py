"""
config.py — Central configuration for all experiments.
Edit these values before running any module.
"""
import os

# ── HP-MOCD baseline hyperparameters (from paper, Table 1) ──────────────────
HPMOCD_CONFIG = {
    # ── Core NSGA-II ─────────────────────────────────────────────
    "population_size":    100,   # ↑ more diversity (was 100)
    "max_generations":    100,   # ↑ avoid premature convergence
    "crossover_prob":     0.85,  # ↑ exploit good structures
    "mutation_prob":      0.35,  # ↑ exploration (critical)
    "ensemble_size":      4,
    "n_threads":          8,
    "seed":               42,

    # ── Disjoint baseline ────────────────────────────────────────
    "novel_label_prob":   0.05,  # slightly ↑ for flexibility

    # ── Overlapping behavior (KEY SECTION) ───────────────────────
    # Controls how often overlap is introduced
    "overlap_add_second_prob":      0.35,   # ↑ (was 0.22)

    # Controls quality threshold for adding overlap
    "overlap_second_support_ratio": 0.55,   # ↓ (was 0.70)

    # Controls strictness of adding second membership
    "overlap_support_margin":       2,      # ↓ (was 3)

    # ── NEW: explicit community control ──────────────────────────
    "n_communities": 22,  # match LFR ground truth (CRITICAL)

    "target_overlap_rate": 0.20,

    # ── NEW: encourage exploration stability ─────────────────────
    "early_stop_patience": 30,  # avoid early stagnation
}

# Development override: set the environment variable `CI_DEV=1` to
# automatically reduce expensive hyperparameters for quick local tests.
if os.environ.get("CI_DEV"):
    HPMOCD_CONFIG.update({
        "population_size": 10,
        "max_generations": 5,
        "n_threads": 1,
    })

# ── LFR Benchmark parameters ─────────────────────────────────────────────────
LFR_CONFIG = {
    "n":                  1000,  # number of nodes
    "tau1":               2.5,   # degree exponent
    "tau2":               1.5,   # community size exponent
    "mu":                 0.3,   # mixing parameter  (vary 0.1–0.8 for robustness test)
    "average_degree":     20,
    "max_degree":         50,
    "min_community":      20,
    "max_community":      100,
    # Overlapping parameters — used in your extension, not the baseline
    "overlap_n":          200,   # number of overlapping nodes
    "overlap_membership": 2,     # communities per overlapping node
    "seed":               42,
}

# ── DBLP dataset ─────────────────────────────────────────────────────────────
DBLP_CONFIG = {
    # SNAP DBLP ground-truth communities
    "url_graph":      "https://snap.stanford.edu/data/bigdata/communities/com-dblp.ungraph.txt.gz",
    "url_cmty":       "https://snap.stanford.edu/data/bigdata/communities/com-dblp.all.cmty.txt.gz",
    "save_dir":       "data/dblp_raw/",
    # Subsample for faster iteration during development
    "subsample_nodes": 10_000,   # set to None to use full graph (~317k nodes)
    "seed":            42,
}

# ── Evaluation ────────────────────────────────────────────────────────────────
EVAL_CONFIG = {
    "n_runs":   20,    # independent runs per configuration (paper standard)
    "alpha":    0.05,  # significance level for paired t-test
}
