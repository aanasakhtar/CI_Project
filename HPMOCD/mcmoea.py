"""
mcmoea.py
---------
Multi-Community Multi-Objective Evolutionary Algorithm (MCMOEA).

This is a pure-Python overlapping community detector built to match the
project's existing algorithm interface:

    partition, runtime = run_mcmoea(G)

The returned partition is always list[frozenset], so run_experiment.py,
metrics.py, and the membership JSON writer can use it unchanged.
"""

from __future__ import annotations

import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Hashable

import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import HPMOCD_CONFIG

Node = Hashable
Individual = dict[Node, set[int]]
Fitness = tuple[float, float, float]


def _community_cover(individual: Individual) -> list[frozenset]:
    communities: dict[int, set[Node]] = {}
    for node, labels in individual.items():
        for label in labels:
            communities.setdefault(label, set()).add(node)
    return [frozenset(nodes) for _label, nodes in sorted(communities.items()) if nodes]


def _hard_partition(individual: Individual, neighbors: dict[Node, list[Node]]) -> list[frozenset]:
    chosen: dict[Node, int] = {}
    for node, labels in individual.items():
        if len(labels) == 1:
            chosen[node] = next(iter(labels))
            continue

        support = Counter()
        for nbr in neighbors[node]:
            support.update(individual[nbr])
        chosen[node] = max(labels, key=lambda label: (support[label], -label))

    groups: dict[int, set[Node]] = {}
    for node, label in chosen.items():
        groups.setdefault(label, set()).add(node)
    return [frozenset(nodes) for _label, nodes in sorted(groups.items()) if nodes]


def _signature(individual: Individual) -> tuple:
    return tuple((node, tuple(sorted(labels))) for node, labels in sorted(individual.items(), key=lambda x: str(x[0])))


def _dominates(a: Fitness, b: Fitness) -> bool:
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def _fast_non_dominated_sort(fitnesses: list[Fitness]) -> list[list[int]]:
    n = len(fitnesses)
    dominated_by_count = [0] * n
    dominates = [[] for _ in range(n)]
    fronts = [[]]

    for p in range(n):
        for q in range(n):
            if _dominates(fitnesses[p], fitnesses[q]):
                dominates[p].append(q)
            elif _dominates(fitnesses[q], fitnesses[p]):
                dominated_by_count[p] += 1
        if dominated_by_count[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in dominates[p]:
                dominated_by_count[q] -= 1
                if dominated_by_count[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    return fronts[:-1]


def _crowding_distance(fitnesses: list[Fitness], front: list[int]) -> dict[int, float]:
    if not front:
        return {}

    distances = {idx: 0.0 for idx in front}
    for obj in range(len(fitnesses[0])):
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


class MCMOEA:
    """
    Overlapping evolutionary community detection.

    Objectives are minimized:
      1. 1 - modularity of a deterministic hard projection
      2. 1 - edge membership agreement
      3. overlap complexity gap from a small target overlap rate
    """

    def __init__(
        self,
        G: nx.Graph,
        cfg: dict = HPMOCD_CONFIG,
        max_memberships: int = 2,
        n_communities: int | None = None,
    ):
        self.G = G
        self.cfg = cfg
        self.nodes = list(G.nodes())
        self.pop_size = int(cfg.get("population_size", 100))
        self.max_gen = int(cfg.get("max_generations", 100))
        self.crossover_prob = float(cfg.get("crossover_prob", 0.8))
        self.mutation_prob = float(cfg.get("mutation_prob", 0.25))
        self.max_memberships = max(1, int(max_memberships))
        self.seed = int(cfg.get("seed", 42))
        self.rng = random.Random(self.seed)

        default_labels = max(2, int(len(self.nodes) ** 0.5 // 2))
        self.n_communities = max(2, min(int(n_communities or default_labels), max(2, len(self.nodes))))
        self.max_label_id = max(self.n_communities * 4, len(self.nodes))
        self.target_overlap_rate = min(0.30, max(0.02, cfg.get("overlap_n", 100) / max(1, len(self.nodes))))

        self.neighbors = {node: list(G.neighbors(node)) for node in self.nodes}
        self.fitness_cache: dict[tuple, Fitness] = {}

    def _seed_from_label_propagation(self) -> Individual:
        try:
            communities = list(nx.community.asyn_lpa_communities(self.G, seed=self.seed))
        except Exception:
            communities = [set(c) for c in nx.connected_components(self.G)]

        individual: Individual = {}
        for cid, community in enumerate(communities):
            for node in community:
                individual[node] = {cid}

        for node in self.nodes:
            individual.setdefault(node, {self.rng.randrange(self.n_communities)})

        return individual

    def _random_individual(self) -> Individual:
        return {node: {self.rng.randrange(self.n_communities)} for node in self.nodes}

    def _add_supported_secondary(self, individual: Individual, node: Node) -> None:
        if len(individual[node]) >= self.max_memberships:
            return

        counts = Counter()
        for nbr in self.neighbors[node]:
            counts.update(individual[nbr])
        if not counts:
            return

        current = individual[node]
        current_best = max((counts.get(label, 0) for label in current), default=0)
        for label, count in counts.most_common():
            if label not in current and count >= max(2, current_best):
                individual[node].add(label)
                return

    def _initial_population(self) -> list[Individual]:
        seed = self._seed_from_label_propagation()
        population = [seed]

        while len(population) < self.pop_size:
            if self.rng.random() < 0.75:
                child = {node: set(labels) for node, labels in seed.items()}
                edits = max(1, len(self.nodes) // 80)
                for _ in range(edits):
                    node = self.rng.choice(self.nodes)
                    if self.rng.random() < 0.55:
                        self._move_to_neighbor_majority(child, node)
                    else:
                        self._add_supported_secondary(child, node)
                population.append(child)
            else:
                population.append(self._random_individual())

        return population

    def _edge_membership_agreement(self, individual: Individual) -> float:
        m = self.G.number_of_edges()
        if m == 0:
            return 0.0

        total = 0.0
        for u, v in self.G.edges():
            lu = individual[u]
            lv = individual[v]
            total += len(lu & lv) / max(1, len(lu | lv))
        return total / m

    def _evaluate(self, individual: Individual) -> Fitness:
        sig = _signature(individual)
        cached = self.fitness_cache.get(sig)
        if cached is not None:
            return cached

        hard = _hard_partition(individual, self.neighbors)
        try:
            modularity = nx.community.modularity(self.G, hard)
        except Exception:
            modularity = -1.0

        edge_agreement = self._edge_membership_agreement(individual)
        overlap_nodes = sum(1 for labels in individual.values() if len(labels) > 1)
        extra_memberships = sum(len(labels) - 1 for labels in individual.values())
        n = max(1, len(individual))
        overlap_rate = overlap_nodes / n
        complexity = abs(overlap_rate - self.target_overlap_rate) + 0.25 * (extra_memberships / n)

        fit = (1.0 - modularity, 1.0 - edge_agreement, complexity)
        self.fitness_cache[sig] = fit
        return fit

    def _move_to_neighbor_majority(self, individual: Individual, node: Node) -> None:
        counts = Counter()
        for nbr in self.neighbors[node]:
            counts.update(individual[nbr])
        if counts:
            individual[node] = {counts.most_common(1)[0][0]}

    def _remove_weak_secondary(self, individual: Individual, node: Node) -> None:
        if len(individual[node]) <= 1:
            return

        counts = Counter()
        for nbr in self.neighbors[node]:
            counts.update(individual[nbr])
        weakest = min(individual[node], key=lambda label: counts.get(label, 0))
        individual[node].discard(weakest)
        if not individual[node]:
            individual[node].add(weakest)

    def _split_large_label(self, individual: Individual) -> None:
        memberships: dict[int, list[Node]] = {}
        for node, labels in individual.items():
            for label in labels:
                memberships.setdefault(label, []).append(node)
        if not memberships:
            return

        label, members = max(memberships.items(), key=lambda item: len(item[1]))
        if len(members) < 10:
            return

        new_label = self.max_label_id + 1
        self.max_label_id = new_label
        self.rng.shuffle(members)
        for node in members[: max(2, len(members) // 5)]:
            individual[node].discard(label)
            individual[node].add(new_label)

    def _crossover(self, p1: Individual, p2: Individual) -> Individual:
        child: Individual = {}
        for node in self.nodes:
            candidates = p1[node] | p2[node]
            if not candidates:
                child[node] = set(p1[node])
                continue

            counts = Counter()
            for nbr in self.neighbors[node]:
                counts.update(p1[nbr])
                counts.update(p2[nbr])

            ranked = sorted(candidates, key=lambda label: (counts[label], self.rng.random()), reverse=True)
            selected = {ranked[0]}
            if self.max_memberships > 1 and len(ranked) > 1:
                if counts[ranked[1]] >= 2 and counts[ranked[1]] >= 0.75 * max(1, counts[ranked[0]]):
                    selected.add(ranked[1])
            child[node] = selected
        return child

    def _mutate(self, individual: Individual) -> Individual:
        child = {node: set(labels) for node, labels in individual.items()}
        if self.rng.random() >= self.mutation_prob:
            return child

        node = self.rng.choice(self.nodes)
        op = self.rng.random()
        if op < 0.45:
            self._move_to_neighbor_majority(child, node)
        elif op < 0.70:
            self._add_supported_secondary(child, node)
        elif op < 0.90:
            self._remove_weak_secondary(child, node)
        else:
            self._split_large_label(child)

        return child

    def _make_child(self, p1: Individual, p2: Individual) -> Individual:
        if self.rng.random() < self.crossover_prob:
            child = self._crossover(p1, p2)
        else:
            child = {node: set(labels) for node, labels in p1.items()}
        return self._mutate(child)

    def _select_next(
        self,
        individuals: list[Individual],
        fitnesses: list[Fitness],
    ) -> tuple[list[Individual], list[Fitness]]:
        unique: dict[tuple, tuple[Individual, Fitness]] = {}
        for ind, fit in zip(individuals, fitnesses):
            sig = _signature(ind)
            old = unique.get(sig)
            if old is None or _dominates(fit, old[1]):
                unique[sig] = (ind, fit)

        inds = [item[0] for item in unique.values()]
        fits = [item[1] for item in unique.values()]
        while len(inds) < self.pop_size:
            rnd = self._random_individual()
            inds.append(rnd)
            fits.append(self._evaluate(rnd))

        next_inds: list[Individual] = []
        next_fits: list[Fitness] = []
        for front in _fast_non_dominated_sort(fits):
            if len(next_inds) + len(front) <= self.pop_size:
                next_inds.extend(inds[i] for i in front)
                next_fits.extend(fits[i] for i in front)
                continue

            remaining = self.pop_size - len(next_inds)
            crowding = _crowding_distance(fits, front)
            ranked = sorted(front, key=lambda i: crowding[i], reverse=True)
            picked = ranked[:remaining]
            next_inds.extend(inds[i] for i in picked)
            next_fits.extend(fits[i] for i in picked)
            break

        return next_inds, next_fits

    def _select_final(self, population: list[Individual], fitnesses: list[Fitness]) -> Individual:
        fronts = _fast_non_dominated_sort(fitnesses)
        first_front = fronts[0] if fronts else list(range(len(population)))

        def score(idx: int) -> float:
            f_mod, f_edge, f_complexity = fitnesses[idx]
            return -f_mod - 0.8 * f_edge - 1.5 * f_complexity

        return population[max(first_front, key=score)]

    def run(self) -> list[frozenset]:
        population = self._initial_population()
        fitnesses = [self._evaluate(ind) for ind in population]

        print(
            f"[MCMOEA] Starting | pop={self.pop_size} | gen={self.max_gen} | "
            f"labels~{self.n_communities} | max_memberships={self.max_memberships}"
        )

        for gen in range(self.max_gen):
            offspring = []
            for _ in range(self.pop_size):
                p1, p2 = self.rng.sample(population, 2)
                offspring.append(self._make_child(p1, p2))

            offspring_fit = [self._evaluate(ind) for ind in offspring]
            population, fitnesses = self._select_next(population + offspring, fitnesses + offspring_fit)

            if (gen + 1) % 10 == 0:
                best = min(fitnesses, key=lambda fit: fit[0] + 0.8 * fit[1] + 1.5 * fit[2])
                print(
                    f"  gen {gen + 1:3d}/{self.max_gen} | "
                    f"f_mod={best[0]:.4f} f_edge={best[1]:.4f} f_complexity={best[2]:.4f}"
                )

        best = self._select_final(population, fitnesses)
        partition = _community_cover(best)
        overlapping_nodes = sum(1 for labels in best.values() if len(labels) > 1)
        print(f"[MCMOEA] Done | communities={len(partition)} | overlapping_nodes={overlapping_nodes}")
        return partition


def run_mcmoea(
    G: nx.Graph,
    cfg: dict = HPMOCD_CONFIG,
    max_memberships: int = 2,
    n_communities: int | None = None,
) -> tuple[list[frozenset], float]:
    t0 = time.perf_counter()
    model = MCMOEA(
        G,
        cfg=cfg,
        max_memberships=max_memberships,
        n_communities=n_communities,
    )
    partition = model.run()
    runtime = time.perf_counter() - t0
    print(f"[MCMOEA] communities={len(partition)}, runtime={runtime:.2f}s")
    return partition, runtime


if __name__ == "__main__":
    from data.load_lfr import load_lfr_overlapping

    graph, truth = load_lfr_overlapping()
    result, elapsed = run_mcmoea(graph)
    print(f"Detected {len(result)} communities in {elapsed:.2f}s")
    print(f"Ground truth communities: {len(truth)}")
