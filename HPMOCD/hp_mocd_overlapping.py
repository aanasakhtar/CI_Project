"""
hp_mocd_overlapping.py
══════════════════════
Overlapping community detection with a true multi-objective NSGA-II design.

This implementation is intentionally pure Python so it can be evolved by teams
without Rust dependencies. It optimizes three explicit objectives:
  1) Modularity quality (maximize Q)
  2) Overlap quality      (maximize edge/membership consistency)
  3) Membership sparsity  (minimize extra memberships)

All objectives are handled directly by NSGA-II (no hard reward/penalty logic).
"""

from __future__ import annotations

import random
import sys
import time
from collections import Counter
from pathlib import Path

import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import HPMOCD_CONFIG
from HPMOCD.hp_mocd_baseline import run_minimal_nsgaii


# ── Representation helpers ───────────────────────────────────────────────────

def _resolve_n_communities(G: nx.Graph, cfg: dict, n_communities: int | None) -> int:
    if n_communities is None:
        n_communities = cfg.get("n_communities")
    if n_communities is None:
        n_communities = max(2, int(len(G) ** 0.5 // 2))
    return max(2, min(int(n_communities), len(G)))


def _partition_to_frozensets(partition: dict[int, set[int]]) -> list[frozenset]:
    grouped: dict[int, set[int]] = {}
    for node, labels in partition.items():
        for label in labels:
            grouped.setdefault(label, set()).add(node)
    return [frozenset(nodes) for _label, nodes in sorted(grouped.items()) if nodes]


def _frozensets_to_partition(communities: list[frozenset]) -> dict[int, set[int]]:
    partition: dict[int, set[int]] = {}
    for cid, comm in enumerate(communities):
        for node in comm:
            partition.setdefault(int(node), set()).add(cid)
    return partition


def _random_partition(G: nx.Graph, n_communities: int) -> dict[int, set[int]]:
    return {int(node): {random.randint(0, n_communities - 1)} for node in G.nodes()}


def _partition_signature(partition: dict[int, set[int]]) -> tuple:
    return tuple((node, tuple(sorted(labels))) for node, labels in sorted(partition.items()))


def _hard_partition_from_overlapping(
    partition: dict[int, set[int]],
    neighbors_cache: dict[int, list[int]],
) -> list[frozenset]:
    """Project overlapping partition to a disjoint partition for modularity.

    Node keeps the label with highest neighbor support.
    """
    disjoint: dict[int, int] = {}
    for node, labels in partition.items():
        if len(labels) == 1:
            disjoint[node] = next(iter(labels))
            continue

        support = {}
        for label in labels:
            s = 0
            for nbr in neighbors_cache[node]:
                if label in partition[nbr]:
                    s += 1
            support[label] = s
        disjoint[node] = max(labels, key=lambda label: (support[label], -label))

    grouped: dict[int, set[int]] = {}
    for node, label in disjoint.items():
        grouped.setdefault(label, set()).add(node)
    return [frozenset(nodes) for _cid, nodes in sorted(grouped.items())]


# ── Objectives ───────────────────────────────────────────────────────────────

def _modularity_objective(
    G: nx.Graph,
    partition: dict[int, set[int]],
    neighbors_cache: dict[int, list[int]],
) -> float:
    hard_partition = _hard_partition_from_overlapping(partition, neighbors_cache)
    q = nx.community.modularity(G, hard_partition)
    return 1.0 - float(q)


def _edge_membership_agreement(G: nx.Graph, partition: dict[int, set[int]]) -> float:
    m = G.number_of_edges()
    if m == 0:
        return 0.0

    total = 0.0
    for u, v in G.edges():
        lu = partition[u]
        lv = partition[v]
        inter = len(lu & lv)
        union = len(lu | lv)
        total += (inter / union) if union > 0 else 0.0
    return total / m


def _overlap_support_quality(
    partition: dict[int, set[int]],
    neighbors_cache: dict[int, list[int]],
) -> float:
    """Quality of overlap memberships based on local neighborhood support."""
    qualities: list[float] = []

    for node, labels in partition.items():
        if len(labels) <= 1:
            continue

        nbrs = neighbors_cache[node]
        if not nbrs:
            continue

        label_scores = []
        for label in labels:
            support = sum(1 for nbr in nbrs if label in partition[nbr])
            label_scores.append(support / len(nbrs))

        if label_scores:
            qualities.append(sum(label_scores) / len(label_scores))

    if not qualities:
        return 0.0
    return sum(qualities) / len(qualities)


def _sparsity_objective(partition: dict[int, set[int]]) -> float:
    n = max(1, len(partition))
    extra = sum(max(0, len(labels) - 1) for labels in partition.values())
    return extra / n


def _overlap_target_gap(partition: dict[int, set[int]], target_overlap_rate: float) -> float:
    n = max(1, len(partition))
    overlap_nodes = sum(1 for labels in partition.values() if len(labels) > 1)
    overlap_rate = overlap_nodes / n
    return abs(overlap_rate - target_overlap_rate)


def _evaluate_objectives(
    G: nx.Graph,
    partition: dict[int, set[int]],
    neighbors_cache: dict[int, list[int]],
    target_overlap_rate: float,
) -> tuple[float, float, float, float]:
    """Return minimization objectives:

    (f_modularity, f_overlap_quality, f_sparsity, f_target_overlap_gap)
    """
    f_mod = _modularity_objective(G, partition, neighbors_cache)

    edge_agree = _edge_membership_agreement(G, partition)
    support_quality = _overlap_support_quality(partition, neighbors_cache)
    target_gap = _overlap_target_gap(partition, target_overlap_rate)

    overlap_quality = 0.6 * edge_agree + 0.4 * support_quality
    f_overlap = 1.0 - overlap_quality

    f_sparse = _sparsity_objective(partition)
    return (f_mod, f_overlap, f_sparse, target_gap)


# ── Operators ────────────────────────────────────────────────────────────────

def _topology_aware_crossover(
    parent1: dict[int, set[int]],
    parent2: dict[int, set[int]],
    neighbors_cache: dict[int, list[int]],
    max_memberships: int,
    add_second_prob: float,
    second_support_ratio: float,
) -> dict[int, set[int]]:
    child: dict[int, set[int]] = {}

    for node, nbrs in neighbors_cache.items():
        candidates = parent1[node] | parent2[node]
        if not candidates:
            child[node] = set(parent1[node])
            continue

        support: dict[int, int] = {label: 0 for label in candidates}
        for nbr in nbrs:
            for label in candidates:
                if label in parent1[nbr] or label in parent2[nbr]:
                    support[label] += 1

        ranked = sorted(candidates, key=lambda label: (support[label], random.random()), reverse=True)
        selected = {ranked[0]}

        if max_memberships > 1 and len(ranked) > 1:
            p = ranked[0]
            s = ranked[1]
            if support[s] >= 2 and support[s] >= second_support_ratio * max(1, support[p]):
                if random.random() < add_second_prob:
                    selected.add(s)

        child[node] = selected

    return child


def _op_boundary_reassign(
    partition: dict[int, set[int]],
    node: int,
    neighbors_cache: dict[int, list[int]],
) -> None:
    nbrs = neighbors_cache[node]
    if not nbrs:
        return

    counts = Counter()
    for nbr in nbrs:
        counts.update(partition[nbr])
    if not counts:
        return

    best = counts.most_common(1)[0][0]
    partition[node] = {best}


def _op_add_secondary(
    partition: dict[int, set[int]],
    node: int,
    neighbors_cache: dict[int, list[int]],
    max_memberships: int,
    support_margin: int,
    target_overlap_rate: float,
) -> None:
    if len(partition[node]) >= max_memberships:
        return

    current_overlap = sum(1 for labels in partition.values() if len(labels) > 1) / max(1, len(partition))
    if current_overlap > target_overlap_rate * 1.5:
        return

    nbrs = neighbors_cache[node]
    if not nbrs:
        return

    counts = Counter()
    for nbr in nbrs:
        counts.update(partition[nbr])
    if not counts:
        return

    current = partition[node]
    current_support = max((counts.get(label, 0) for label in current), default=0)
    for label, c in counts.most_common():
        if label not in current and c >= current_support + support_margin:
            partition[node] = set(current) | {label}
            return


def _op_remove_secondary(
    partition: dict[int, set[int]],
    node: int,
    neighbors_cache: dict[int, list[int]],
) -> None:
    labels = partition[node]
    if len(labels) <= 1:
        return

    nbrs = neighbors_cache[node]
    if not nbrs:
        partition[node] = {next(iter(labels))}
        return

    counts = Counter()
    for nbr in nbrs:
        counts.update(partition[nbr])

    weakest = min(labels, key=lambda label: counts.get(label, 0))
    updated = set(labels)
    updated.discard(weakest)
    if updated:
        partition[node] = updated


def _op_split_community(partition: dict[int, set[int]], max_label_id: int) -> None:
    memberships: dict[int, list[int]] = {}
    for node, labels in partition.items():
        for label in labels:
            memberships.setdefault(label, []).append(node)
    if not memberships:
        return

    largest_label, members = max(memberships.items(), key=lambda item: len(item[1]))
    if len(members) < 8:
        return

    new_label = max_label_id + 1
    subset = members[: max(2, len(members) // 4)]
    for node in subset:
        partition[node].discard(largest_label)
        partition[node].add(new_label)


def _op_merge_communities(partition: dict[int, set[int]]) -> None:
    memberships: dict[int, set[int]] = {}
    for node, labels in partition.items():
        for label in labels:
            memberships.setdefault(label, set()).add(node)

    labels = list(memberships.keys())
    if len(labels) < 2:
        return

    a, b = random.sample(labels, 2)
    inter = len(memberships[a] & memberships[b])
    union = max(1, len(memberships[a] | memberships[b]))
    if inter / union < 0.25:
        return

    keep, drop = (a, b) if len(memberships[a]) >= len(memberships[b]) else (b, a)
    for node, node_labels in partition.items():
        if drop in node_labels:
            node_labels.discard(drop)
            node_labels.add(keep)


def _mutate(
    partition: dict[int, set[int]],
    neighbors_cache: dict[int, list[int]],
    mutation_prob: float,
    max_memberships: int,
    support_margin: int,
    target_overlap_rate: float,
    max_label_id: int,
) -> dict[int, set[int]]:
    child = {node: set(labels) for node, labels in partition.items()}
    if random.random() >= mutation_prob:
        return child

    node = random.choice(list(child.keys()))
    op = random.random()

    if op < 0.35:
        _op_boundary_reassign(child, node, neighbors_cache)
    elif op < 0.60:
        _op_add_secondary(child, node, neighbors_cache, max_memberships, support_margin, target_overlap_rate)
    elif op < 0.80:
        _op_remove_secondary(child, node, neighbors_cache)
    elif op < 0.90:
        _op_split_community(child, max_label_id)
    else:
        _op_merge_communities(child)

    # safety: no node should become unlabeled
    for n, labels in child.items():
        if not labels:
            child[n] = {partition[n] and next(iter(partition[n])) or 0}

    return child


# ── NSGA-II core ─────────────────────────────────────────────────────────────

class OverlappingNSGAII:
    def __init__(
        self,
        G: nx.Graph,
        cfg: dict = HPMOCD_CONFIG,
        max_memberships: int = 2,
        n_communities: int | None = None,
        seed_partition: list[frozenset] | None = None,
    ):
        self.G = G
        self.cfg = cfg
        self.pop_size = int(cfg["population_size"])
        self.max_gen = int(cfg["max_generations"])
        self.crossover_prob = float(cfg["crossover_prob"])
        self.mutation_prob = float(cfg["mutation_prob"])

        self.max_memberships = max_memberships
        self.n_communities = _resolve_n_communities(G, cfg, n_communities)
        self.max_label_id = max(self.n_communities * 4, len(G) - 1)

        self.target_overlap_rate = min(0.95, max(0.01, cfg.get("overlap_n", 100) / max(1, len(G))))
        self.add_second_prob = float(cfg.get("overlap_add_second_prob", 0.10))
        self.second_support_ratio = float(cfg.get("overlap_second_support_ratio", 0.90))
        self.support_margin = int(cfg.get("overlap_support_margin", 3))

        self.neighbors_cache = {int(node): list(G.neighbors(node)) for node in G.nodes()}
        self.fitness_cache: dict[tuple, tuple[float, float, float, float]] = {}
        self._external_seed_partition = seed_partition

    @staticmethod
    def _dominates(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
        return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))

    def _fast_non_dominated_sort(self, fitnesses: list[tuple[float, ...]]) -> list[list[int]]:
        n = len(fitnesses)
        dominates = [[] for _ in range(n)]
        dominated_count = [0] * n
        fronts = [[]]

        for p in range(n):
            for q in range(n):
                if self._dominates(fitnesses[p], fitnesses[q]):
                    dominates[p].append(q)
                elif self._dominates(fitnesses[q], fitnesses[p]):
                    dominated_count[p] += 1
            if dominated_count[p] == 0:
                fronts[0].append(p)

        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in dominates[p]:
                    dominated_count[q] -= 1
                    if dominated_count[q] == 0:
                        next_front.append(q)
            i += 1
            fronts.append(next_front)

        return fronts[:-1]

    def _crowding_distance(
        self,
        fitnesses: list[tuple[float, ...]],
        front: list[int],
    ) -> dict[int, float]:
        if not front:
            return {}

        distances = {idx: 0.0 for idx in front}
        n_obj = len(fitnesses[0])
        for obj in range(n_obj):
            ordered = sorted(front, key=lambda idx: fitnesses[idx][obj])
            distances[ordered[0]] = float("inf")
            distances[ordered[-1]] = float("inf")

            lo = fitnesses[ordered[0]][obj]
            hi = fitnesses[ordered[-1]][obj]
            span = hi - lo
            if span == 0:
                continue

            for i in range(1, len(ordered) - 1):
                prev_val = fitnesses[ordered[i - 1]][obj]
                next_val = fitnesses[ordered[i + 1]][obj]
                distances[ordered[i]] += (next_val - prev_val) / span

        return distances

    def _evaluate(self, individual: dict[int, set[int]]) -> tuple[float, float, float, float]:
        signature = _partition_signature(individual)
        cached = self.fitness_cache.get(signature)
        if cached is not None:
            return cached

        fit = _evaluate_objectives(
            self.G,
            individual,
            self.neighbors_cache,
            target_overlap_rate=self.target_overlap_rate,
        )
        self.fitness_cache[signature] = fit
        return fit

    def _seed_population(self) -> list[dict[int, set[int]]]:
        if self._external_seed_partition is not None:
            seed_cover = _frozensets_to_partition(self._external_seed_partition)
        else:
            seed_cover = _frozensets_to_partition(run_minimal_nsgaii(self.G, cfg=self.cfg)[0])

        population: list[dict[int, set[int]]] = [seed_cover]
        while len(population) < self.pop_size:
            if random.random() < 0.70:
                ind = {node: set(labels) for node, labels in seed_cover.items()}
                for _ in range(max(1, len(self.G) // 100)):
                    node = random.choice(list(ind.keys()))
                    _op_boundary_reassign(ind, node, self.neighbors_cache)
                population.append(ind)
            else:
                population.append(_random_partition(self.G, self.n_communities))

        return population

    def _filter_duplicates(
        self,
        individuals: list[dict[int, set[int]]],
        fitnesses: list[tuple[float, ...]],
    ) -> tuple[list[dict[int, set[int]]], list[tuple[float, ...]]]:
        seen: dict[tuple, int] = {}
        unique_inds: list[dict[int, set[int]]] = []
        unique_fit: list[tuple[float, ...]] = []

        for ind, fit in zip(individuals, fitnesses):
            sig = _partition_signature(ind)
            idx = seen.get(sig)
            if idx is None:
                seen[sig] = len(unique_inds)
                unique_inds.append(ind)
                unique_fit.append(fit)
            else:
                # Keep the dominating/better fit if duplicate signature appears.
                current = unique_fit[idx]
                if self._dominates(fit, current):
                    unique_inds[idx] = ind
                    unique_fit[idx] = fit

        return unique_inds, unique_fit

    def _next_population(
        self,
        individuals: list[dict[int, set[int]]],
        fitnesses: list[tuple[float, ...]],
    ) -> tuple[list[dict[int, set[int]]], list[tuple[float, ...]]]:
        fronts = self._fast_non_dominated_sort(fitnesses)
        new_inds: list[dict[int, set[int]]] = []
        new_fit: list[tuple[float, ...]] = []

        for front in fronts:
            if len(new_inds) + len(front) <= self.pop_size:
                new_inds.extend(individuals[i] for i in front)
                new_fit.extend(fitnesses[i] for i in front)
                continue

            remaining = self.pop_size - len(new_inds)
            crowd = self._crowding_distance(fitnesses, front)
            ranked = sorted(front, key=lambda i: crowd[i], reverse=True)
            chosen = ranked[:remaining]
            new_inds.extend(individuals[i] for i in chosen)
            new_fit.extend(fitnesses[i] for i in chosen)
            break

        return new_inds, new_fit

    def _make_child(self, p1: dict[int, set[int]], p2: dict[int, set[int]]) -> dict[int, set[int]]:
        if random.random() < self.crossover_prob:
            child = _topology_aware_crossover(
                p1,
                p2,
                self.neighbors_cache,
                self.max_memberships,
                self.add_second_prob,
                self.second_support_ratio,
            )
        else:
            child = {node: set(labels) for node, labels in p1.items()}

        child = _mutate(
            child,
            self.neighbors_cache,
            self.mutation_prob,
            self.max_memberships,
            self.support_margin,
            self.target_overlap_rate,
            self.max_label_id,
        )
        return child

    def _select_final(self, population: list[dict[int, set[int]]], fitnesses: list[tuple[float, ...]]) -> dict[int, set[int]]:
        fronts = self._fast_non_dominated_sort(fitnesses)
        first_front = fronts[0] if fronts else list(range(len(population)))

        # Balanced tie-break inside first Pareto front.
        def score(i: int) -> float:
            f_mod, f_ov, f_sp, f_gap = fitnesses[i]
            return -f_mod - 0.9 * f_ov - 0.6 * f_sp - 5.0 * f_gap

        best_idx = max(first_front, key=score)
        return population[best_idx]

    def run(self) -> list[frozenset]:
        population = self._seed_population()
        fitnesses = [self._evaluate(ind) for ind in population]

        print(
            f"[OverlappingNSGAII] Starting | pop={self.pop_size} | gen={self.max_gen} | "
            f"initial_labels~{self.n_communities} | max_memberships={self.max_memberships}"
        )

        for gen in range(self.max_gen):
            offspring = []
            for _ in range(self.pop_size):
                p1, p2 = random.sample(population, 2)
                offspring.append(self._make_child(p1, p2))

            offspring_fit = [self._evaluate(ind) for ind in offspring]
            combined = population + offspring
            combined_fit = fitnesses + offspring_fit
            combined, combined_fit = self._filter_duplicates(combined, combined_fit)

            while len(combined) < self.pop_size:
                rnd = _random_partition(self.G, self.n_communities)
                combined.append(rnd)
                combined_fit.append(self._evaluate(rnd))

            population, fitnesses = self._next_population(combined, combined_fit)

            if (gen + 1) % 10 == 0:
                best = min(fitnesses, key=lambda f: f[0] + 0.9 * f[1] + 0.6 * f[2] + 5.0 * f[3])
                print(
                    f"  gen {gen+1:3d}/{self.max_gen} | "
                    f"f_mod={best[0]:.4f} f_ov={best[1]:.4f} f_sp={best[2]:.4f} f_gap={best[3]:.4f}"
                )

        best_partition = self._select_final(population, fitnesses)
        result = _partition_to_frozensets(best_partition)
        overlap_nodes = sum(1 for labels in best_partition.values() if len(labels) > 1)

        print(
            f"[OverlappingNSGAII] Done | communities={len(result)} | "
            f"overlapping_nodes={overlap_nodes}"
        )
        return result


def run_hp_mocd_overlapping(
    G: nx.Graph,
    cfg: dict = HPMOCD_CONFIG,
    max_memberships: int = 2,
    n_communities: int | None = None,
    seed_partition: list[frozenset] | None = None,
) -> tuple[list[frozenset], float]:
    t0 = time.perf_counter()
    model = OverlappingNSGAII(
        G,
        cfg=cfg,
        max_memberships=max_memberships,
        n_communities=n_communities,
        seed_partition=seed_partition,
    )
    best = model.run()
    runtime = time.perf_counter() - t0
    print(f"[HP-MOCD Overlapping] communities={len(best)}, runtime={runtime:.2f}s")
    return best, runtime


if __name__ == "__main__":
    sys.path.insert(0, "..")
    from data.load_lfr import load_lfr_overlapping

    G, gt = load_lfr_overlapping()
    pred, rt = run_hp_mocd_overlapping(G)
    print(f"Detected {len(pred)} communities in {rt:.2f}s")
    print(f"Ground truth communities: {len(gt)}")
