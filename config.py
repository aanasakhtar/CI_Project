"""
config.py — Central configuration for all experiments.
Edit these values before running any module.
"""
import os

# ── HP-MOCD baseline hyperparameters (from paper, Table 1) ──────────────────
HPMOCD_CONFIG = {
    "population_size":    100,
    "max_generations":    100,
    "crossover_prob":     0.8,
    "mutation_prob":      0.25,
    "ensemble_size":      4,     # parents per crossover
    "n_threads":          8,     # set to your core count
    "seed":               42,
    # Disjoint baseline tuning
    "novel_label_prob":   0.03,
    # Overlap extension tuning
    "overlap_add_second_prob":      0.22,
    "overlap_second_support_ratio": 0.70,
    "overlap_support_margin":       3,
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
    "overlap_n":          100,   # number of overlapping nodes
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
