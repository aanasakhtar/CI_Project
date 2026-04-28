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


def _seed_partition_lpa(G: nx.Graph, seed: int) -> dict[int, int]:
    """Build a deterministic disjoint seed partition using async label propagation."""
    communities = list(nx.community.asyn_lpa_communities(G, seed=seed))
    partition: dict[int, int] = {}
    for cid, comm in enumerate(communities):
        for node in comm:
            partition[node] = cid
    return partition


def _jitter_partition(
    partition: dict[int, int],
    G: nx.Graph,
    n_communities: int,
    jitter_rate: float,
) -> dict[int, int]:
    import random

    child = dict(partition)
    for node in G.nodes():
        if random.random() < jitter_rate:
            child[node] = random.randint(0, n_communities - 1)
    return child


def _neighbor_cache(G: nx.Graph) -> dict[int, list[int]]:
    return {node: list(G.neighbors(node)) for node in G.nodes()}


def _community_counts(partition: dict[int, int], neighbors: list[int]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for nbr in neighbors:
        cid = partition[nbr]
        counts[cid] = counts.get(cid, 0) + 1
    return counts


def _topology_aware_crossover(
    parent1: dict[int, int],
    parent2: dict[int, int],
    neighbors_cache: dict[int, list[int]],
) -> dict[int, int]:
    import random

    child: dict[int, int] = {}
    for node, nbrs in neighbors_cache.items():
        labels = {parent1[node], parent2[node]}
        if len(labels) == 1:
            child[node] = parent1[node]
            continue

        support1 = _community_counts(parent1, nbrs)
        support2 = _community_counts(parent2, nbrs)

        def support(label: int) -> int:
            return max(support1.get(label, 0), support2.get(label, 0))

        ranked = sorted(labels, key=lambda label: (support(label), random.random()), reverse=True)
        child[node] = ranked[0]
    return child


def _topology_aware_mutation(
    partition: dict[int, int],
    G: nx.Graph,
    neighbors_cache: dict[int, list[int]],
    mutation_prob: float,
    max_label_id: int,
    novel_label_prob: float,
) -> dict[int, int]:
    import random

    new_partition = dict(partition)
    if random.random() >= mutation_prob:
        return new_partition

    node = random.choice(list(G.nodes()))
    nbrs = neighbors_cache.get(node, [])
    if not nbrs:
        return new_partition

    counts = _community_counts(partition, nbrs)
    if not counts:
        return new_partition

    current = new_partition[node]
    candidate = max(counts.items(), key=lambda item: (item[1], random.random()))[0]
    if candidate != current:
        new_partition[node] = candidate

    current_support = counts.get(current, 0)
    if random.random() < novel_label_prob and current_support <= 1:
        used_labels = set(new_partition.values())
        fresh = None
        for label in range(max_label_id + 1):
            if label not in used_labels:
                fresh = label
                break
        if fresh is not None:
            new_partition[node] = fresh
    return new_partition


def _crowding_distance(fitnesses: list[tuple[float, float]], front: list[int]) -> dict[int, float]:
    if not front:
        return {}

    distances = {idx: 0.0 for idx in front}
    for objective in range(2):
        ordered = sorted(front, key=lambda idx: fitnesses[idx][objective])
        distances[ordered[0]] = float("inf")
        distances[ordered[-1]] = float("inf")

        min_val = fitnesses[ordered[0]][objective]
        max_val = fitnesses[ordered[-1]][objective]
        span = max_val - min_val
        if span == 0:
            continue

        for i in range(1, len(ordered) - 1):
            prev_val = fitnesses[ordered[i - 1]][objective]
            next_val = fitnesses[ordered[i + 1]][objective]
            distances[ordered[i]] += (next_val - prev_val) / span

    return distances


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
        self.mutation_prob = cfg["mutation_prob"]
        self.novel_label_prob = cfg.get("novel_label_prob", 0.03)
        self.neighbors_cache = _neighbor_cache(G)
        self.seed_partition = _seed_partition_lpa(self.G, cfg.get("seed", 42))
        seeded_count = max(2, len(set(self.seed_partition.values())))
        heuristic_count = max(2, int(len(self.G) ** 0.5 // 2))
        self.n_communities = max(cfg.get("n_communities", 0), seeded_count, heuristic_count)
        self.max_label_id = max(self.n_communities * 3, len(self.G) - 1)

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

    def _select_next_population(
        self,
        combined: list[dict[int, int]],
        combined_fit: list[tuple[float, float]],
    ) -> tuple[list[dict[int, int]], list[tuple[float, float]]]:
        fronts = self._fast_non_dominated_sort(combined_fit)
        new_pop: list[dict[int, int]] = []
        new_fits: list[tuple[float, float]] = []

        for front in fronts:
            if len(new_pop) + len(front) <= self.pop_size:
                new_pop.extend(combined[i] for i in front)
                new_fits.extend(combined_fit[i] for i in front)
                continue

            remaining = self.pop_size - len(new_pop)
            crowd = _crowding_distance(combined_fit, front)
            ranked = sorted(front, key=lambda idx: (crowd[idx], -combined_fit[idx][0] - combined_fit[idx][1]), reverse=True)
            picked = ranked[:remaining]
            new_pop.extend(combined[i] for i in picked)
            new_fits.extend(combined_fit[i] for i in picked)
            break

        return new_pop, new_fits

    def run(self) -> list[frozenset]:
        import random
        pop: list[dict[int, int]] = [dict(self.seed_partition)]
        while len(pop) < self.pop_size:
            if random.random() < 0.65:
                pop.append(_jitter_partition(self.seed_partition, self.G, self.n_communities, jitter_rate=0.08))
            else:
                pop.append(_random_partition(self.G, self.n_communities))
        fits = [_evaluate(self.G, p) for p in pop]
        best_q = max(1 - f1 - f2 for f1, f2 in fits)
        stall = 0
        patience = max(10, self.max_gen // 5)

        for gen in range(self.max_gen):
            offspring = []
            for _ in range(self.pop_size):
                p1, p2 = random.sample(pop, 2)
                if random.random() < self.cfg["crossover_prob"]:
                    child = _topology_aware_crossover(p1, p2, self.neighbors_cache)
                else:
                    child = dict(p1)

                child = _topology_aware_mutation(
                    child,
                    self.G,
                    self.neighbors_cache,
                    self.mutation_prob,
                    self.max_label_id,
                    self.novel_label_prob,
                )
                offspring.append(child)

            off_fits = [_evaluate(self.G, p) for p in offspring]
            combined = pop + offspring
            combined_fit = fits + off_fits

            new_pop, new_fits = self._select_next_population(combined, combined_fit)
            pop, fits = new_pop, new_fits

            current_q = max(1 - f1 - f2 for f1, f2 in fits)
            if current_q > best_q:
                best_q = current_q
                stall = 0
            else:
                stall += 1
            if stall >= patience:
                break

        # Select best via Q(C) = 1 - f1 - f2
        best_idx = max(range(len(fits)), key=lambda i: 1 - fits[i][0] - fits[i][1])
        best_partition = pop[best_idx]

        communities: dict[int, set] = {}
        for node, cid in best_partition.items():
            communities.setdefault(cid, set()).add(node)
        return [frozenset(c) for c in communities.values()]


def run_minimal_nsgaii(
    G: nx.Graph,
    cfg: dict = HPMOCD_CONFIG,
) -> tuple[list[frozenset], float]:
    """Run the stronger local NSGA-II baseline and return partition + runtime."""
    t0 = time.perf_counter()
    best = MinimalNSGAII(G, cfg=cfg).run()
    runtime = time.perf_counter() - t0
    print(f"[MinimalNSGAII] communities={len(best)}, runtime={runtime:.2f}s")
    return best, runtime


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from data.load_lfr import load_lfr_disjoint

    G, gt = load_lfr_disjoint()

    best, _, rt = run_hp_mocd(G)

    print(f"Detected {len(best)} communities in {rt:.2f}s")
    print(f"Ground truth: {len(gt)} communities")
