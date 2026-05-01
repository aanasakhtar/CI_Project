"""hp_mocd_overlapping.py (v3 - overlap-aware)

Overlapping community detection as a multi-objective NSGA-II extension of
HP-MOCD.

Root cause of v2 failure (0 overlapping nodes):
    v2 removed f_gap entirely, eliminating pressure toward overlap. Pure
    structural objectives are satisfied by a disjoint partition so the
    algorithm converged to disjoint solutions. Operator thresholds were
    adjusted in v3 to encourage overlap.

Three-objective design (v3):
    f1: 1 - modularity (structure quality)
    f2: mean conductance (community tightness)
    f3: overlap quality loss (rewards well-supported overlap, penalises
            zero or unsupported overlap). Range [0,1].

Operator fixes:
    - add_secondary probability raised to 40% (was 30%)
    - thresholds relaxed (k_min reduced, support ratio relaxed)
    - remove_secondary reduced to 15% (was 25%)

Interface (unchanged):
        from HPMOCD.hp_mocd_overlapping import run_hp_mocd_overlapping
        partition, runtime = run_hp_mocd_overlapping(G, max_memberships=2)
"""

from __future__ import annotations

import random
import sys
import time
from collections import Counter, deque
from pathlib import Path

import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import HPMOCD_CONFIG
from HPMOCD.hp_mocd_baseline import run_minimal_nsgaii


# SECTION 1 - REPRESENTATION HELPERS (unchanged from v1)

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
    """Project overlapping partition → disjoint (node keeps best-supported label)."""
    disjoint: dict[int, int] = {}
    for node, labels in partition.items():
        if len(labels) == 1:
            disjoint[node] = next(iter(labels))
            continue
        support = {}
        for label in labels:
            support[label] = sum(1 for nbr in neighbors_cache[node] if label in partition[nbr])
        disjoint[node] = max(labels, key=lambda label: (support[label], -label))

    grouped: dict[int, set[int]] = {}
    for node, label in disjoint.items():
        grouped.setdefault(label, set()).add(node)
    return [frozenset(nodes) for _cid, nodes in sorted(grouped.items())]


# SECTION 2 - OBJECTIVES (redesigned)

def _modularity_objective(
    G: nx.Graph,
    partition: dict[int, set[int]],
    neighbors_cache: dict[int, list[int]],
) -> float:
    """f1 = 1 − Q  (minimise).  Uses hard projection, same as v1."""
    hard = _hard_partition_from_overlapping(partition, neighbors_cache)
    try:
        q = nx.community.modularity(G, hard)
    except Exception:
        q = -1.0
    return 1.0 - float(q)


def _conductance_objective(
    G: nx.Graph,
    partition: dict[int, set[int]],
    neighbors_cache: dict[int, list[int]],
) -> float:
    """
    f2 = mean conductance over all communities  (minimise → tight communities).

    Conductance of community C:
        φ(C) = cut(C) / min(vol(C), vol(V-C))

    where cut = edges crossing boundary, vol = sum of degrees inside C.

    Range: [0, 1].  0 = perfectly isolated community, 1 = all edges cross.

    Why this is better than the old edge-agreement proxy
    ─────────────────────────────────────────────────────
    Edge-agreement counts intra-community edge fraction, which correlates
    strongly with community SIZE — large communities always look better.
    Conductance normalises by volume so small tight communities are not
    penalised, making it a fairer, more discriminating quality signal.
    """
    hard = _hard_partition_from_overlapping(partition, neighbors_cache)
    if not hard:
        return 1.0

    total_vol = 2 * G.number_of_edges()
    if total_vol == 0:
        return 1.0

    conductances: list[float] = []
    for comm in hard:
        vol_in = sum(G.degree(v) for v in comm)
        cut = sum(
            1 for v in comm for nbr in G.neighbors(v) if nbr not in comm
        )
        denominator = min(vol_in, total_vol - vol_in)
        if denominator <= 0:
            conductances.append(0.0)
        else:
            conductances.append(cut / denominator)

    return sum(conductances) / max(1, len(conductances))


def _overlap_quality_objective(
    partition: dict[int, set[int]],
    neighbors_cache: dict[int, list[int]],
    target_overlap_rate: float = 0.20,
    support_threshold: float = 0.20,
) -> float:
    """
    f3 = overlap quality loss  (minimise).

    Composed of two terms that TOGETHER create the right gradient:

    (a) no_overlap_penalty  [0, 1]
        = max(0,  target_rate − actual_rate) / target_rate
        A disjoint solution (0 overlap) gets full penalty = 1.0 here.
        Once actual_rate >= target_rate this term goes to 0.
        This is a SOFT floor — not a hard count constraint like old f_gap.

    (b) unsupported_overlap_penalty  [0, 1]
        For each overlapping node: ratio = min_support / max_support.
        Penalises memberships where ratio < support_threshold.
        A node with 5 neighbours in comm A and 4 in comm B → ratio=0.8 → no penalty.
        A node with 5 neighbours in comm A and 0 in comm B → ratio=0.0 → penalised.

    f3 = 0.5 * no_overlap_penalty + 0.5 * unsupported_overlap_penalty

    Why this works where v2 failed
    ───────────────────────────────
    v2's f3 only penalised BAD overlap; a disjoint partition had f3=0 (no
    overlap, no penalty → no pressure to add overlap).  This f3 explicitly
    penalises TOO LITTLE overlap via term (a), so disjoint solutions are no
    longer free riders.  Term (b) then guides WHERE overlap is placed.
    """
    n = max(1, len(partition))
    overlap_nodes = sum(1 for labels in partition.values() if len(labels) > 1)
    actual_rate = overlap_nodes / n

    # (a) Penalise too little overlap (soft floor).
    if target_overlap_rate > 0:
        no_overlap_penalty = max(0.0, (target_overlap_rate - actual_rate) / target_overlap_rate)
    else:
        no_overlap_penalty = 0.0

    # (b) Penalise unsupported overlap nodes.
    unsupported_penalties: list[float] = []
    for node, labels in partition.items():
        if len(labels) <= 1:
            continue
        nbrs = neighbors_cache[node]
        if not nbrs:
            unsupported_penalties.append(1.0)
            continue

        label_support = {
            label: sum(1 for nbr in nbrs if label in partition[nbr])
            for label in labels
        }
        max_sup = max(label_support.values())
        min_sup = min(label_support.values())

        if max_sup == 0:
            unsupported_penalties.append(1.0)
            continue

        ratio = min_sup / max_sup
        if ratio < support_threshold:
            unsupported_penalties.append(support_threshold - ratio)

    if unsupported_penalties:
        unsupported_penalty = min(1.0, sum(unsupported_penalties) / n)
    else:
        unsupported_penalty = 0.0

    return 0.5 * no_overlap_penalty + 0.5 * unsupported_penalty


def _evaluate_objectives(
    G: nx.Graph,
    partition: dict[int, set[int]],
    neighbors_cache: dict[int, list[int]],
    target_overlap_rate: float = 0.20,
) -> tuple[float, float, float]:
    """
    Return (f1, f2, f3) — all minimised.

    f1: 1 − modularity       (structure quality via hard projection)
    f2: mean conductance      (community tightness via hard projection)
    f3: overlap quality loss  (soft overlap floor + unsupported-overlap penalty)
    """
    f1 = _modularity_objective(G, partition, neighbors_cache)
    f2 = _conductance_objective(G, partition, neighbors_cache)
    f3 = _overlap_quality_objective(
        partition, neighbors_cache,
        target_overlap_rate=target_overlap_rate,
    )
    return (f1, f2, f3)


# SECTION 3 - OPERATORS (rebalanced + guided add)

def _op_boundary_reassign(
    partition: dict[int, set[int]],
    node: int,
    neighbors_cache: dict[int, list[int]],
) -> None:
    """Assign node to majority-vote community among its neighbours (disjoint)."""
    nbrs = neighbors_cache[node]
    if not nbrs:
        return
    counts: Counter = Counter()
    for nbr in nbrs:
        counts.update(partition[nbr])
    if counts:
        best = counts.most_common(1)[0][0]
        partition[node] = {best}


def _op_guided_add_secondary(
    partition: dict[int, set[int]],
    node: int,
    neighbors_cache: dict[int, list[int]],
    max_memberships: int,
    k_min: int = 1,
    support_ratio_threshold: float = 0.20,
) -> None:
    """
    Add a secondary community membership where structurally supported.

    Thresholds (v3, relaxed from v2):
      k_min = 1           — only 1 neighbour needed in the second community
                            (v2 used 2, which was too strict for sparse nodes)
      support_ratio = 0.20 — secondary support must be ≥ 20 % of primary
                            (v2 used 0.30 — rarely triggered)

    The f3 objective then EVALUATES whether the added membership is truly
    supported (ratio ≥ 0.20) or weak (penalised).  The operator's job is
    to PROPOSE; the objective's job is to FILTER via selection pressure.
    """
    if len(partition[node]) >= max_memberships:
        return

    nbrs = neighbors_cache[node]
    if not nbrs:
        return

    counts: Counter = Counter()
    for nbr in nbrs:
        counts.update(partition[nbr])

    current = partition[node]
    primary_support = max((counts.get(label, 0) for label in current), default=0)

    for label, count in counts.most_common():
        if label in current:
            continue
        if count >= k_min and count >= support_ratio_threshold * max(1, primary_support):
            partition[node] = set(current) | {label}
            return


def _op_remove_weak_secondary(
    partition: dict[int, set[int]],
    node: int,
    neighbors_cache: dict[int, list[int]],
) -> None:
    """Remove the least-supported secondary membership."""
    labels = partition[node]
    if len(labels) <= 1:
        return

    nbrs = neighbors_cache[node]
    if not nbrs:
        # No neighbours — keep only the smallest label (deterministic).
        partition[node] = {min(labels)}
        return

    counts: Counter = Counter()
    for nbr in nbrs:
        counts.update(partition[nbr])

    weakest = min(labels, key=lambda label: counts.get(label, 0))
    updated = set(labels)
    updated.discard(weakest)
    if updated:
        partition[node] = updated


def _op_split_community(partition: dict[int, set[int]], max_label_id: int) -> int:
    """
    Split the largest community by peeling off a small fraction.

    Returns the new max_label_id.
    """
    memberships: dict[int, list[int]] = {}
    for node, labels in partition.items():
        for label in labels:
            memberships.setdefault(label, []).append(node)
    if not memberships:
        return max_label_id

    largest_label, members = max(memberships.items(), key=lambda item: len(item[1]))
    if len(members) < 8:
        return max_label_id

    new_label = max_label_id + 1
    # Peel off ≤ 20 % of the community (reduced from v1's 25 %).
    subset = members[: max(2, len(members) // 5)]
    for node in subset:
        partition[node].discard(largest_label)
        partition[node].add(new_label)
    return new_label


def _op_merge_communities(partition: dict[int, set[int]]) -> None:
    """
    Merge two communities only when they overlap significantly (Jaccard ≥ 0.25).

    Same guard as v1 — reduced application probability handles the rest.
    """
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
    max_label_id: int,
    k_min: int = 1,
    support_ratio_threshold: float = 0.20,
) -> tuple[dict[int, set[int]], int]:
    """
    Apply one mutation operator to a copy of `partition`.

    Operator probabilities (v3):
        boundary_reassign    : 38 %   — good structural move, keep dominant
        guided_add_secondary : 40 %   — raised from 30 %; primary driver of overlap
        remove_weak_secondary: 15 %   — reduced from 25 %; was pruning too much
        split_community      :  5 %   — kept small
        merge_communities    :  2 %   — minimal; too destructive

    The key change from v2: add (40 %) >> remove (15 %).  In v2 they were
    balanced (30/25), which meant the algorithm added and removed overlap at
    nearly the same rate, producing no net drift toward overlap.

    Returns
    -------
    mutated partition, (possibly updated) max_label_id
    """
    child = {node: set(labels) for node, labels in partition.items()}
    if random.random() >= mutation_prob:
        return child, max_label_id

    node = random.choice(list(child.keys()))
    op = random.random()

    if op < 0.38:
        _op_boundary_reassign(child, node, neighbors_cache)
    elif op < 0.78:          # 0.38 + 0.40
        _op_guided_add_secondary(
            child, node, neighbors_cache, max_memberships,
            k_min=k_min, support_ratio_threshold=support_ratio_threshold,
        )
    elif op < 0.93:          # + 0.15
        _op_remove_weak_secondary(child, node, neighbors_cache)
    elif op < 0.98:          # + 0.05
        max_label_id = _op_split_community(child, max_label_id)
    else:                    # + 0.02
        _op_merge_communities(child)

    # Safety: no node may become unlabelled.
    for n, labels in child.items():
        if not labels:
            child[n] = set(partition[n]) or {0}

    return child, max_label_id


# SECTION 4 - CROSSOVER (identical logic to v1, kept for reference)

def _topology_aware_crossover(
    parent1: dict[int, set[int]],
    parent2: dict[int, set[int]],
    neighbors_cache: dict[int, list[int]],
    max_memberships: int,
    add_second_prob: float = 0.35,
    second_support_ratio: float = 0.55,
) -> dict[int, set[int]]:
    """
    For each node, rank candidate labels by combined neighbourhood support
    across both parents.  Select the top label; optionally add the second
    when its support clears both an absolute (≥ 2) and relative threshold.
    """
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
            p_sup = support[ranked[0]]
            s_sup = support[ranked[1]]
            if s_sup >= 2 and s_sup >= second_support_ratio * max(1, p_sup):
                if random.random() < add_second_prob:
                    selected.add(ranked[1])

        child[node] = selected

    return child


# SECTION 5 - NSGA-II CORE

class OverlappingNSGAII:
    """
    NSGA-II–based overlapping community detector.

    Improvements over v1
    ────────────────────
    • 3 objectives instead of 4 (f_gap removed)
    • Conductance replaces edge-agreement as f2
    • Structural overlap penalty replaces sparsity + f_gap as f3
    • Diversified seeding (35/35/30 split)
    • Rebalanced mutation operators
    • Adaptive stagnation recovery
    """

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

        # Guided-add hyper-parameters (relaxed in v3).
        self.k_min = 1
        self.support_ratio_threshold = 0.20

        # Crossover second-label parameters.
        self.add_second_prob = float(cfg.get("overlap_add_second_prob", 0.35))
        self.second_support_ratio = float(cfg.get("overlap_second_support_ratio", 0.55))

        # Overlap target passed through to f3.
        self.target_overlap_rate = float(
            cfg.get("target_overlap_rate",
                    cfg.get("overlap_n", 200) / max(1, len(G)))
        )

        # Stagnation recovery.
        self.patience = int(cfg.get("early_stop_patience", 20))
        self._stagnation_boost_factor = 1.5   # temporary mutation rate multiplier

        self.neighbors_cache: dict[int, list[int]] = {
            int(node): list(G.neighbors(node)) for node in G.nodes()
        }
        self.fitness_cache: dict[tuple, tuple[float, float, float]] = {}
        self._external_seed_partition = seed_partition

    # -- Dominance / sorting

    @staticmethod
    def _dominates(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
        return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))

    def _fast_non_dominated_sort(
        self, fitnesses: list[tuple[float, ...]]
    ) -> list[list[int]]:
        n = len(fitnesses)
        dominates = [[] for _ in range(n)]
        dominated_count = [0] * n
        fronts: list[list[int]] = [[]]

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
            next_front: list[int] = []
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
                distances[ordered[i]] += (
                    fitnesses[ordered[i + 1]][obj] - fitnesses[ordered[i - 1]][obj]
                ) / span
        return distances

    # -- Fitness evaluation

    def _evaluate(self, individual: dict[int, set[int]]) -> tuple[float, float, float]:
        sig = _partition_signature(individual)
        cached = self.fitness_cache.get(sig)
        if cached is not None:
            return cached
        fit = _evaluate_objectives(
            self.G, individual, self.neighbors_cache,
            target_overlap_rate=self.target_overlap_rate,
        )
        self.fitness_cache[sig] = fit
        return fit

    # -- Population initialisation (diversified seeding)

    def _guided_overlap_injection(
        self, base: dict[int, set[int]]
    ) -> dict[int, set[int]]:
        """
        Take the disjoint HP-MOCD seed and add secondary memberships only
        at nodes with strong structural support (k_min neighbours in second
        community, support_ratio ≥ threshold).
        """
        ind = {node: set(labels) for node, labels in base.items()}
        node_list = list(ind.keys())
        random.shuffle(node_list)
        for node in node_list:
            _op_guided_add_secondary(
                ind, node, self.neighbors_cache, self.max_memberships,
                k_min=self.k_min,
                support_ratio_threshold=self.support_ratio_threshold,
            )
        return ind

    def _lpa_seed(self, resolution_jitter: float = 0.0) -> dict[int, set[int]]:
        """LPA seed with optional label jitter for diversity."""
        try:
            communities = list(nx.community.asyn_lpa_communities(
                self.G, seed=random.randint(0, 9999)
            ))
        except Exception:
            communities = [set(c) for c in nx.connected_components(self.G)]
        ind: dict[int, set[int]] = {}
        for cid, comm in enumerate(communities):
            for node in comm:
                ind[int(node)] = {cid}
        # Fill any missing nodes.
        for node in self.G.nodes():
            ind.setdefault(int(node), {random.randint(0, self.n_communities - 1)})
        if resolution_jitter > 0:
            n_jitter = max(1, int(len(ind) * resolution_jitter))
            for node in random.sample(list(ind.keys()), n_jitter):
                ind[node] = {random.randint(0, self.n_communities - 1)}
        return ind

    def _seed_population(self) -> list[dict[int, set[int]]]:
        # --- Obtain the base disjoint HP-MOCD seed. ---
        if self._external_seed_partition is not None:
            seed_disjoint = _frozensets_to_partition(self._external_seed_partition)
        else:
            seed_disjoint = _frozensets_to_partition(
                run_minimal_nsgaii(self.G, cfg=self.cfg)[0]
            )

        population: list[dict[int, set[int]]] = []

        # 35 % — exact HP-MOCD seed (disjoint).
        n_exact = max(1, int(0.35 * self.pop_size))
        population.append(seed_disjoint)
        while len(population) < n_exact:
            ind = {node: set(labels) for node, labels in seed_disjoint.items()}
            # Small boundary perturbation only.
            for _ in range(max(1, len(self.G) // 100)):
                node = random.choice(list(ind.keys()))
                _op_boundary_reassign(ind, node, self.neighbors_cache)
            population.append(ind)

        # 35 % — topology-guided overlap injection from HP-MOCD seed.
        n_overlap = max(1, int(0.35 * self.pop_size))
        while len(population) < n_exact + n_overlap:
            population.append(self._guided_overlap_injection(seed_disjoint))

        # 30 % — diverse LPA seeds at different resolutions.
        while len(population) < self.pop_size:
            jitter = random.uniform(0.05, 0.20)
            if random.random() < 0.5:
                population.append(self._lpa_seed(resolution_jitter=jitter))
            else:
                population.append(_random_partition(self.G, self.n_communities))

        return population[:self.pop_size]

    # ── Offspring generation ─────────────────────────────────────────────────

    def _make_child(
        self,
        p1: dict[int, set[int]],
        p2: dict[int, set[int]],
        mutation_prob_override: float | None = None,
    ) -> dict[int, set[int]]:
        if random.random() < self.crossover_prob:
            child = _topology_aware_crossover(
                p1, p2,
                self.neighbors_cache,
                self.max_memberships,
                self.add_second_prob,
                self.second_support_ratio,
            )
        else:
            child = {node: set(labels) for node, labels in p1.items()}

        mut_prob = mutation_prob_override if mutation_prob_override is not None else self.mutation_prob
        child, self.max_label_id = _mutate(
            child,
            self.neighbors_cache,
            mut_prob,
            self.max_memberships,
            self.max_label_id,
            k_min=self.k_min,
            support_ratio_threshold=self.support_ratio_threshold,
        )
        return child

    # ── Population management ────────────────────────────────────────────────

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
                if self._dominates(fit, unique_fit[idx]):
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

    def _inject_diversity(self, population: list[dict[int, set[int]]]) -> list[dict[int, set[int]]]:
        """Replace the worst 20 % of the population with fresh individuals."""
        n_inject = max(1, int(0.20 * self.pop_size))
        fresh = []
        for _ in range(n_inject):
            if random.random() < 0.5:
                fresh.append(self._lpa_seed(resolution_jitter=random.uniform(0.10, 0.25)))
            else:
                fresh.append(_random_partition(self.G, self.n_communities))
        # Replace tail of population (NSGA-II already orders worst last).
        return population[: self.pop_size - n_inject] + fresh

    # ── Final solution selection ─────────────────────────────────────────────

    def _select_final(
        self,
        population: list[dict[int, set[int]]],
        fitnesses: list[tuple[float, ...]],
    ) -> dict[int, set[int]]:
        fronts = self._fast_non_dominated_sort(fitnesses)
        first_front = fronts[0] if fronts else list(range(len(population)))

        def score(i: int) -> float:
            f1, f2, f3 = fitnesses[i]
            # Structure (f1) dominates; overlap quality (f3) weighted to
            # prefer solutions with good overlap over purely disjoint ones;
            # conductance (f2) as secondary tiebreak.
            return -f1 - 0.4 * f2 - 0.6 * f3

        best_idx = max(first_front, key=score)
        return population[best_idx]

    # ── Main loop ────────────────────────────────────────────────────────────

    def run(self, return_history: bool = False) -> list[frozenset] | tuple[list[frozenset], list[dict[str, float]]]:
        population = self._seed_population()
        fitnesses = [self._evaluate(ind) for ind in population]
        history: list[dict[str, float]] = []

        # ══════════════════════════════════════════════════════════════════
        # TWO-PHASE EVOLUTION
        # ══════════════════════════════════════════════════════════════════
        #
        # Phase 1 (first 40 % of generations): DISJOINT ONLY.
        #   Overlap operators disabled by setting effective_max_memberships=1.
        #   Only f1 (modularity) + f2 (conductance) drive evolution.
        #   Goal: reach a strong disjoint partition before introducing overlap.
        #
        # Phase 2 (remaining 60 %): OVERLAP ALLOWED.
        #   All operators and objectives active.  Starting from a good
        #   disjoint base means overlap is added into solid structure.
        # ══════════════════════════════════════════════════════════════════

        phase1_gens = max(10, int(0.40 * self.max_gen))

        def _composite(fit: tuple) -> float:
            return fit[0] + 0.4 * fit[1] + 0.6 * fit[2]

        best_composite = min(_composite(f) for f in fitnesses)
        stagnation_window: deque[float] = deque([best_composite], maxlen=self.patience)
        consecutive_no_improve = 0

        print(
            f"[OverlappingNSGAII-v3] Starting | "
            f"pop={self.pop_size} | gen={self.max_gen} | "
            f"communities~{self.n_communities} | max_memberships={self.max_memberships} | "
            f"target_overlap={self.target_overlap_rate:.0%} | "
            f"phase1={phase1_gens}gen / phase2={self.max_gen - phase1_gens}gen"
        )

        for gen in range(self.max_gen):
            in_phase1 = gen < phase1_gens

            # ── Stagnation check (phase 2 only) ──────────────────────────
            current_best = min(_composite(f) for f in fitnesses)
            if current_best < best_composite - 1e-5:
                best_composite = current_best
                consecutive_no_improve = 0
            else:
                consecutive_no_improve += 1
            stagnation_window.append(current_best)

            is_stagnant = (
                not in_phase1
                and consecutive_no_improve >= self.patience
                and (max(stagnation_window) - min(stagnation_window)) < 1e-4
            )
            if is_stagnant:
                population = self._inject_diversity(population)
                fitnesses = [self._evaluate(ind) for ind in population]
                consecutive_no_improve = 0
                print(f"  gen {gen+1:3d} | [STAGNATION] diversity injected")

            # ── Offspring generation ──────────────────────────────────────
            mut_prob = (
                min(1.0, self.mutation_prob * self._stagnation_boost_factor)
                if is_stagnant else self.mutation_prob
            )
            effective_max_mem = 1 if in_phase1 else self.max_memberships

            offspring: list[dict[int, set[int]]] = []
            for _ in range(self.pop_size):
                p1, p2 = random.sample(population, 2)
                if random.random() < self.crossover_prob:
                    child = _topology_aware_crossover(
                        p1, p2, self.neighbors_cache,
                        effective_max_mem,
                        self.add_second_prob, self.second_support_ratio,
                    )
                else:
                    child = {node: set(labels) for node, labels in p1.items()}
                child, self.max_label_id = _mutate(
                    child, self.neighbors_cache, mut_prob,
                    effective_max_mem, self.max_label_id,
                    k_min=self.k_min,
                    support_ratio_threshold=self.support_ratio_threshold,
                )
                offspring.append(child)

            offspring_fit = [self._evaluate(ind) for ind in offspring]
            combined = population + offspring
            combined_fit = fitnesses + offspring_fit
            combined, combined_fit = self._filter_duplicates(combined, combined_fit)
            while len(combined) < self.pop_size:
                rnd = _random_partition(self.G, self.n_communities)
                combined.append(rnd)
                combined_fit.append(self._evaluate(rnd))
            population, fitnesses = self._next_population(combined, combined_fit)

            scores = [_composite(fit) for fit in fitnesses]
            best_score = min(scores)
            avg_score = sum(scores) / max(1, len(scores))
            best_so_far = best_score if not history else min(history[-1]["best_so_far"], best_score)
            history.append({
                "generation": float(gen + 1),
                "avg_fitness": float(avg_score),
                "best_fitness": float(best_score),
                "best_so_far": float(best_so_far),
                "phase": 1.0 if in_phase1 else 2.0,
            })

            # ── Phase transition ──────────────────────────────────────────
            if gen + 1 == phase1_gens:
                best_f1_now = min(f[0] for f in fitnesses)
                print(
                    f"  gen {gen+1:3d} | >>> PHASE 2 START "
                    f"(best f_mod={best_f1_now:.4f}) <<<"
                )
                consecutive_no_improve = 0
                best_composite = min(_composite(f) for f in fitnesses)
                stagnation_window.clear()
                stagnation_window.append(best_composite)

            # ── Progress logging ──────────────────────────────────────────
            if (gen + 1) % 10 == 0:
                best_fit = min(fitnesses, key=_composite)
                fronts_now = self._fast_non_dominated_sort(fitnesses)
                best_idx = max(fronts_now[0], key=lambda i: -_composite(fitnesses[i]))
                ovlp_count = sum(
                    1 for labels in population[best_idx].values() if len(labels) > 1
                )
                phase_tag = "P1" if in_phase1 else "P2"
                print(
                    f"  gen {gen+1:3d}/{self.max_gen} [{phase_tag}] | "
                    f"f_mod={best_fit[0]:.4f} "
                    f"f_cond={best_fit[1]:.4f} "
                    f"f_ovlp={best_fit[2]:.4f} | "
                    f"overlap_nodes={ovlp_count}"
                )

        best_partition = self._select_final(population, fitnesses)
        result = _partition_to_frozensets(best_partition)
        overlap_nodes = sum(1 for labels in best_partition.values() if len(labels) > 1)
        print(
            f"[OverlappingNSGAII-v3] Done | "
            f"communities={len(result)} | overlapping_nodes={overlap_nodes}"
        )
        if return_history:
            return result, history
        return result


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC ENTRY POINT  (interface unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def run_hp_mocd_overlapping(
    G: nx.Graph,
    cfg: dict = HPMOCD_CONFIG,
    max_memberships: int = 2,
    n_communities: int | None = None,
    seed_partition: list[frozenset] | None = None,
) -> tuple[list[frozenset], float]:
    """
    Run the improved overlapping HP-MOCD extension and return (partition, runtime).

    Parameters
    ----------
    G               : nx.Graph — input graph
    cfg             : dict     — hyperparameters (from config.py)
    max_memberships : int      — maximum communities per node (default 2)
    n_communities   : int|None — override community count (None = auto)
    seed_partition  : list[frozenset]|None — external disjoint seed (None = HP-MOCD)

    Returns
    -------
    partition : list[frozenset] — overlapping community cover
    runtime   : float           — wall-clock seconds
    """
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
    print(f"[HP-MOCD Overlapping v3] communities={len(best)}, runtime={runtime:.2f}s")
    return best, runtime


def run_hp_mocd_overlapping_with_history(
    G: nx.Graph,
    cfg: dict = HPMOCD_CONFIG,
    max_memberships: int = 2,
    n_communities: int | None = None,
    seed_partition: list[frozenset] | None = None,
) -> tuple[list[frozenset], float, list[dict[str, float]]]:
    t0 = time.perf_counter()
    model = OverlappingNSGAII(
        G,
        cfg=cfg,
        max_memberships=max_memberships,
        n_communities=n_communities,
        seed_partition=seed_partition,
    )
    best, history = model.run(return_history=True)
    runtime = time.perf_counter() - t0
    print(f"[HP-MOCD Overlapping v3] communities={len(best)}, runtime={runtime:.2f}s")
    return best, runtime, history


if __name__ == "__main__":
    sys.path.insert(0, "..")
    from data.load_lfr import load_lfr_overlapping

    G, gt = load_lfr_overlapping()
    pred, rt = run_hp_mocd_overlapping(G)
    print(f"Detected {len(pred)} communities in {rt:.2f}s")
    print(f"Ground truth communities: {len(gt)}")