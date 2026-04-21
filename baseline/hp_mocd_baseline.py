"""
baseline/hp_mocd_baseline.py
────────────────────────────
Thin wrapper around the official `pymocd` Python package.

`pymocd` is the open-source Rust/PyO3 implementation released alongside
the paper. It exposes a NetworkX-compatible API.

Install:  pip install pymocd

Docs:     https://oliveira-sh.github.io/dpymocd/
"""

import time
import sys
from pathlib import Path

import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Official package — install via: pip install pymocd
try:
    import pymocd
    HAS_HPMOCD = True
except ImportError:
    pymocd = None
    HAS_HPMOCD = False
    print("[WARNING] pymocd package not found. "
          "Run: pip install pymocd")

from config import HPMOCD_CONFIG


# ── Main runner ──────────────────────────────────────────────────────────────

def run_hp_mocd(
    G: nx.Graph,
    cfg: dict = HPMOCD_CONFIG,
    return_pareto: bool = False,
) -> tuple[list[frozenset], list[list[frozenset]] | None, float]:
    """
    Run HP-MOCD on graph G and return the best (selected) partition.

    Parameters
    ----------
    G            : nx.Graph — input graph (unweighted, undirected)
    cfg          : dict     — hyperparameters (see config.py)
    return_pareto: bool     — if True, also return full Pareto front

    Returns
    -------
    best_partition : list of frozensets  (selected via Q(C) = 1 - f1 - f2)
    pareto_front   : list of partitions  (only if return_pareto=True, else None)
    runtime_s      : float               (wall-clock seconds)
    """
    if not HAS_HPMOCD:
        print("[HP-MOCD] package unavailable; using MinimalNSGAII fallback.")
        t0 = time.perf_counter()
        best = MinimalNSGAII(G, cfg=cfg).run()
        runtime = time.perf_counter() - t0
        print(f"[HP-MOCD:fallback] communities={len(best)}, runtime={runtime:.2f}s")
        return best, None, runtime

    t0 = time.perf_counter()

    model = pymocd.HpMocd(
        G,
        debug_level=0,
        pop_size=cfg["population_size"],
        num_gens=cfg["max_generations"],
        cross_rate=cfg["crossover_prob"],
        mut_rate=cfg["mutation_prob"],
    )

    result = model.run()

    runtime = time.perf_counter() - t0

    # `run()` returns a node -> community partition dict.
    communities: dict[int, set] = {}
    for node, cid in result.items():
        communities.setdefault(cid, set()).add(node)
    best = [frozenset(c) for c in communities.values()]

    pareto: list[list[frozenset]] | None = None
    if return_pareto and hasattr(model, "generate_pareto_front"):
        front = model.generate_pareto_front()
        pareto = []
        for partition, _objectives in front:
            grouped: dict[int, set] = {}
            for node, cid in partition.items():
                grouped.setdefault(cid, set()).add(node)
            pareto.append([frozenset(c) for c in grouped.values()])

    print(f"[HP-MOCD] communities={len(best)}, runtime={runtime:.2f}s")
    return best, pareto, runtime


# ── Fallback: pure-Python NSGA-II skeleton (if hp-mocd unavailable) ─────────
# Use this to understand the algorithm internals or to prototype your extension.

def _evaluate(G: nx.Graph, partition: dict[int, int]) -> tuple[float, float]:
    """Compute (f1, f2) for a node→community mapping."""
    m = G.number_of_edges()
    if m == 0:
        return 1.0, 1.0

    # Group nodes by community
    communities: dict[int, list] = {}
    for node, cid in partition.items():
        communities.setdefault(cid, []).append(node)

    # f1: fraction of edges NOT internal
    internal = sum(
        1 for u, v in G.edges()
        if partition[u] == partition[v]
    )
    f1 = 1.0 - internal / m

    # f2: sum of squared relative community sizes (by degree)
    deg = dict(G.degree())
    two_m = 2 * m
    f2 = sum(
        (sum(deg[v] for v in members) / two_m) ** 2
        for members in communities.values()
    )
    return f1, f2


def _random_partition(G: nx.Graph, n_communities: int) -> dict[int, int]:
    import random
    nodes = list(G.nodes())
    return {v: random.randint(0, n_communities - 1) for v in nodes}


class MinimalNSGAII:
    """
    Bare-bones NSGA-II for community detection.
    NOT production-quality — use this only to understand / prototype.
    For real experiments, use the hp-mocd package above.
    """
    def __init__(self, G: nx.Graph, cfg: dict = HPMOCD_CONFIG):
        self.G = G
        self.cfg = cfg
        self.pop_size = cfg["population_size"]
        self.max_gen  = cfg["max_generations"]

    def _dominates(self, a: tuple, b: tuple) -> bool:
        return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))

    def _fast_non_dominated_sort(self, fitnesses):
        n = len(fitnesses)
        S = [[] for _ in range(n)]
        dom_count = [0] * n
        fronts = [[]]
        for p in range(n):
            for q in range(n):
                if self._dominates(fitnesses[p], fitnesses[q]):
                    S[p].append(q)
                elif self._dominates(fitnesses[q], fitnesses[p]):
                    dom_count[p] += 1
            if dom_count[p] == 0:
                fronts[0].append(p)
        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in S[p]:
                    dom_count[q] -= 1
                    if dom_count[q] == 0:
                        next_front.append(q)
            i += 1
            fronts.append(next_front)
        return fronts[:-1]

    def run(self) -> list[frozenset]:
        import random
        n_cmty = max(2, int(len(self.G) ** 0.5 // 3))
        pop = [_random_partition(self.G, n_cmty) for _ in range(self.pop_size)]
        fits = [_evaluate(self.G, p) for p in pop]

        for gen in range(self.max_gen):
            # Offspring via simple crossover + mutation
            offspring = []
            for _ in range(self.pop_size):
                p1, p2 = random.sample(pop, 2)
                child = {
                    v: p1[v] if random.random() < 0.5 else p2[v]
                    for v in self.G.nodes()
                }
                # Mutation: reassign to most-common neighbor community
                if random.random() < self.cfg["mutation_prob"]:
                    v = random.choice(list(self.G.nodes()))
                    nbr_cmts = [child[u] for u in self.G.neighbors(v)]
                    if nbr_cmts:
                        child[v] = max(set(nbr_cmts), key=nbr_cmts.count)
                offspring.append(child)

            off_fits = [_evaluate(self.G, p) for p in offspring]
            combined     = pop + offspring
            combined_fit = fits + off_fits

            fronts = self._fast_non_dominated_sort(combined_fit)
            new_pop, new_fits = [], []
            for front in fronts:
                if len(new_pop) + len(front) <= self.pop_size:
                    new_pop  += [combined[i] for i in front]
                    new_fits += [combined_fit[i] for i in front]
                else:
                    # Fill remaining slots (simplified — no crowding distance)
                    needed = self.pop_size - len(new_pop)
                    new_pop  += [combined[i] for i in front[:needed]]
                    new_fits += [combined_fit[i] for i in front[:needed]]
                    break
            pop, fits = new_pop, new_fits

        # Select best via Q(C) = 1 - f1 - f2
        best_idx = max(range(len(fits)), key=lambda i: 1 - fits[i][0] - fits[i][1])
        best_partition = pop[best_idx]

        communities: dict[int, set] = {}
        for node, cid in best_partition.items():
            communities.setdefault(cid, set()).add(node)
        return [frozenset(c) for c in communities.values()]


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from data.load_lfr import load_lfr_disjoint

    G, gt = load_lfr_disjoint()

    best, _, rt = run_hp_mocd(G)

    print(f"Detected {len(best)} communities in {rt:.2f}s")
    print(f"Ground truth: {len(gt)} communities")
