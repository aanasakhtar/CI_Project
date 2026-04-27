"""
hp_mocd_overlapping.py
══════════════════════
Overlapping community detection — faithful extension of HP-MOCD.

Based on the HP-MOCD paper (Santos et al. 2025):
  - Topology-aware crossover: inherit the community with more neighbour support
  - Topology-aware mutation:  adopt the most common community among neighbours
  - NSGA-II loop:             identical to baseline

The ONLY things that change vs MinimalNSGAII:
  1. Representation : {node: int} → {node: set[int]}
  2. Crossover      : topology-aware (HP-MOCD), extended for multi-membership
  3. Mutation       : topology-aware ADDITIVE — adds a second membership
                      instead of replacing, so nodes don't all over-lap
  4. f1 / f2        : adapted for multi-membership (Jaccard + fractional degree)

HOW TO USE
──────────
    from hp_mocd_overlapping import run_hp_mocd_overlapping
    partition, runtime = run_hp_mocd_overlapping(G)
    # partition is list[frozenset] — identical format to baseline
"""

import time
import random
import sys
from pathlib import Path
from collections import Counter

import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from config import HPMOCD_CONFIG
except ImportError:
    HPMOCD_CONFIG = {
        "population_size": 50,
        "max_generations":  100,
        "crossover_prob":   0.8,
        "mutation_prob":    0.1,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — REPRESENTATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _estimate_n_communities(G: nx.Graph) -> int:
    """
    Estimate a reasonable number of communities for the graph.

    The baseline uses  max(2, int(n**0.5 // 3))  which gives only 2 for
    small graphs (e.g. Karate Club with 34 nodes → 2 communities).
    That's too few; real graphs typically have more.

    We use a slightly larger estimate:  max(2, int(n**0.5 // 2)).
    For Karate Club (n=34), this still gives 2.
    For large graphs (n=1000), it gives ~15, which is reasonable.

    IMPORTANT:
    This value is an initial label pool. Current crossover/mutation operators
    do not invent brand-new community IDs, so this also acts as an effective
    upper bound unless you pass a larger override via n_communities.
    """
    n = len(G)
    return max(2, int(n ** 0.5 // 2))


def _resolve_n_communities(
    G: nx.Graph,
    cfg: dict,
    n_communities: int | None,
) -> int:
    """Resolve initial community-label pool from arg -> cfg -> heuristic."""
    if n_communities is None:
        n_communities = cfg.get("n_communities") if isinstance(cfg, dict) else None

    if n_communities is None:
        n_communities = _estimate_n_communities(G)

    resolved = int(n_communities)
    return max(2, min(resolved, len(G)))


def _random_overlapping_partition(
    G: nx.Graph,
    n_communities: int,
) -> dict[int, set[int]]:
    """
    Create a random starting individual.
    Every node starts in exactly ONE community — same as baseline.
    Overlap emerges through crossover and mutation, not forced at init.
    """
    return {node: {random.randint(0, n_communities - 1)} for node in G.nodes()}


def _partition_to_frozensets(partition: dict[int, set[int]]) -> list[frozenset]:
    """
    Convert {node: set[int]} → list[frozenset].
    Nodes in multiple communities appear in multiple frozensets.
    This is the output format — identical to baseline.
    """
    community_members: dict[int, set] = {}
    for node, cids in partition.items():
        for cid in cids:
            community_members.setdefault(cid, set()).add(node)
    return [frozenset(members) for members in community_members.values() if members]


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — OBJECTIVE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _evaluate_overlapping(
    G: nx.Graph,
    partition: dict[int, set[int]],
) -> tuple[float, float]:
    """
    Compute (f1, f2) for an overlapping partition. Both minimised.

    f1 — adapted from HP-MOCD's "fraction of external edges"
         For overlapping: use Jaccard similarity of membership sets per edge.
         shared_strength(u,v) = |mems(u) ∩ mems(v)| / |mems(u) ∪ mems(v)|
         = 1 when both nodes are in identical communities
         = 0 when they share no community
         For disjoint (sets of size 1): reduces to exactly 0 or 1, matching baseline.
         f1 = 1 - mean(shared_strength over all edges)

    f2 — adapted from HP-MOCD's "community size balance"
         For overlapping: a node in k communities contributes degree/k to each.
         This prevents the trivial "everyone in all communities" cheat.
         f2 = sum_c (volume(c) / 2m)^2
         + small overlap penalty to discourage unnecessary memberships
    """
    m = G.number_of_edges()
    if m == 0:
        return 1.0, 1.0

    degree = dict(G.degree())

    # ── f1: Jaccard-based shared strength ────────────────────────────────────
    total_shared = 0.0
    for u, v in G.edges():
        mems_u = partition[u]
        mems_v = partition[v]
        intersection = len(mems_u & mems_v)
        union        = len(mems_u | mems_v)
        total_shared += intersection / union   # union always >= 1

    f1 = 1.0 - (total_shared / m)

    # ── f2: fractional degree balance ────────────────────────────────────────
    all_cids: set[int] = set()
    for cids in partition.values():
        all_cids.update(cids)

    community_volume: dict[int, float] = {cid: 0.0 for cid in all_cids}
    for node, cids in partition.items():
        k = len(cids)
        fractional_deg = degree[node] / k      # split degree across communities
        for cid in cids:
            community_volume[cid] += fractional_deg

    two_m = 2 * m
    f2 = sum((vol / two_m) ** 2 for vol in community_volume.values())

    # Small penalty per extra membership — overlap must earn its keep via f1
    extra_memberships = sum(max(0, len(cids) - 1) for cids in partition.values())
    f2 += 0.05 * (extra_memberships / len(partition))

    return f1, f2

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — TOPOLOGY-AWARE CROSSOVER  (HP-MOCD paper)
# ══════════════════════════════════════════════════════════════════════════════

def _topology_aware_crossover(
    parent1: dict[int, set[int]],
    parent2: dict[int, set[int]],
    max_memberships: int,
    neighbors_cache: dict[int, list],
) -> dict[int, set[int]]:
    """
    HP-MOCD topology-aware crossover, extended for overlapping.

    For each node v:
      - Collect candidate communities from both parents: p1[v] ∪ p2[v]
      - Count how many of v's neighbours belong to each candidate community
        (looking at both parents' assignments for each neighbour)
      - Keep the top max_memberships communities by neighbour support
      - Ties broken by random coin-flip (same as vanilla crossover for tied case)

    This is strictly better than a coin-flip because it uses graph structure
    to decide which community labels are more "natural" for each node.
    """
    child: dict[int, set[int]] = {}

    for node, nbrs in neighbors_cache.items():
        candidates = parent1[node] | parent2[node]

        if len(candidates) == 1:
            # Both parents agree — no decision needed
            child[node] = set(candidates)
            continue

        # Count neighbour support for each candidate community
        support: dict[int, int] = {cid: 0 for cid in candidates}
        for nbr in nbrs:
            for cid in candidates:
                if cid in parent1[nbr] or cid in parent2[nbr]:
                    support[cid] += 1

        # Sort by support, random tiebreak
        ranked = sorted(
            candidates,
            key=lambda c: (support[c], random.random()),
            reverse=True,
        )

        child[node] = set(ranked[:max_memberships])

    return child


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — TOPOLOGY-AWARE MUTATION  (HP-MOCD paper, additive version)
# ══════════════════════════════════════════════════════════════════════════════

def _topology_aware_mutation(
    partition: dict[int, set[int]],
    mutation_prob: float,
    max_memberships: int,
    neighbors_cache: dict[int, list],
    max_label_id: int,
    novel_label_prob: float = 0.10,
) -> dict[int, set[int]]:
    """
    HP-MOCD topology-aware mutation, extended for overlapping.

    HP-MOCD's mutation: reassign a node to its most common neighbour community.
    For overlapping: ADD the most common neighbour community as a second
    membership, but only if the node isn't already in it AND it has room.

    To avoid an artificial cap from initial label count, mutation can also
    introduce a fresh (unused) community id with a small probability.

    Key fix vs previous version: we ADD instead of REPLACE.
    Replacing caused every node to end up in multiple communities because
    the top-k neighbours of every node span multiple communities.
    Adding is conservative — a node only gains a second membership if there
    is clear topological evidence it belongs to another community too.

    With probability mutation_prob, one random node is mutated.
    """
    # Shallow copy outer dict; only mutated node gets a new set
    new_partition = {node: cids for node, cids in partition.items()}

    if random.random() >= mutation_prob:
        return new_partition

    node = random.choice(list(neighbors_cache.keys()))
    nbrs = neighbors_cache[node]

    if not nbrs:
        return new_partition

    # Count community membership across neighbours (HP-MOCD's local majority rule)
    neighbour_counts: Counter = Counter()
    for nbr in nbrs:
        for cid in partition[nbr]:
            neighbour_counts[cid] += 1

    if not neighbour_counts:
        return new_partition

    current_memberships = partition[node]

    # Find the most popular neighbour community that node is NOT yet in
    best_new_community = None
    for cid, _ in neighbour_counts.most_common():
        if cid not in current_memberships:
            best_new_community = cid
            break

    if best_new_community is None:
        if len(current_memberships) >= max_memberships:
            return new_partition

        # No unseen neighbour label found. Occasionally introduce a novel label
        # so the model is not capped by initial n_communities.
        if random.random() < novel_label_prob:
            used_labels = set().union(*partition.values()) if partition else set()
            if len(used_labels) < (max_label_id + 1):
                attempts = 0
                fresh_label = None
                while attempts < 20:
                    candidate = random.randint(0, max_label_id)
                    if candidate not in used_labels:
                        fresh_label = candidate
                        break
                    attempts += 1
                if fresh_label is None:
                    for candidate in range(max_label_id + 1):
                        if candidate not in used_labels:
                            fresh_label = candidate
                            break
                if fresh_label is not None:
                    new_partition[node] = current_memberships | {fresh_label}

        return new_partition

    # Only add the second membership if node has room
    if len(current_memberships) < max_memberships:
        new_partition[node] = current_memberships | {best_new_community}
    else:
        # At capacity — do a swap: replace the weakest membership with the best new one.
        # Weakest = the community with fewest neighbour connections.
        weakest = min(
            current_memberships,
            key=lambda c: neighbour_counts.get(c, 0),
        )
        # Only swap if the new community is actually better supported
        if neighbour_counts.get(best_new_community, 0) > neighbour_counts.get(weakest, 0):
            updated = (current_memberships - {weakest}) | {best_new_community}
            new_partition[node] = updated

    return new_partition


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — OverlappingNSGAII  (NSGA-II loop identical to baseline)
# ══════════════════════════════════════════════════════════════════════════════

class OverlappingNSGAII:
    """
    NSGA-II for overlapping community detection, closely following HP-MOCD.

    What changes vs MinimalNSGAII (baseline):
      • Representation : {node: set[int]} instead of {node: int}
      • Crossover      : topology-aware (HP-MOCD paper)
      • Mutation       : topology-aware, additive
      • f1 / f2        : adapted for multi-membership

    What is IDENTICAL to baseline:
      • Non-dominated sorting
      • Population selection (fill by front)
      • Final selection: Q = 1 - f1 - f2
      • NSGA-II loop structure
    """

    def __init__(
        self,
        G: nx.Graph,
        cfg: dict        = HPMOCD_CONFIG,
        max_memberships: int = 2,
        n_communities: int | None = None,
    ):
        self.G               = G
        self.cfg             = cfg
        self.pop_size        = cfg["population_size"]
        self.max_gen         = cfg["max_generations"]
        self.crossover_prob  = cfg["crossover_prob"]
        self.mutation_prob   = cfg["mutation_prob"]
        self.max_memberships = max_memberships
        self.n_communities   = _resolve_n_communities(G, cfg, n_communities)
        self.max_label_id    = max(0, len(G) - 1)

        # Pre-compute neighbour lists once — mirrors HP-MOCD's hash map optimisation
        self.neighbors_cache: dict[int, list] = {
            node: list(G.neighbors(node)) for node in G.nodes()
        }

    def _dominates(self, a: tuple, b: tuple) -> bool:
        return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))

    def _fast_non_dominated_sort(self, fitnesses: list[tuple]) -> list[list[int]]:
        """Identical to MinimalNSGAII baseline."""
        n = len(fitnesses)
        dominates_list  = [[] for _ in range(n)]
        dominated_count = [0] * n
        fronts = [[]]

        for p in range(n):
            for q in range(n):
                if self._dominates(fitnesses[p], fitnesses[q]):
                    dominates_list[p].append(q)
                elif self._dominates(fitnesses[q], fitnesses[p]):
                    dominated_count[p] += 1
            if dominated_count[p] == 0:
                fronts[0].append(p)

        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in dominates_list[p]:
                    dominated_count[q] -= 1
                    if dominated_count[q] == 0:
                        next_front.append(q)
            i += 1
            fronts.append(next_front)

        return fronts[:-1]

    def run(self) -> list[frozenset]:
        """
        Run overlapping NSGA-II. Loop structure identical to MinimalNSGAII.
        Returns list[frozenset] — same format as baseline.
        """
        # Initialise
        population = [
            _random_overlapping_partition(self.G, self.n_communities)
            for _ in range(self.pop_size)
        ]
        fitnesses = [_evaluate_overlapping(self.G, ind) for ind in population]

        print(
            f"[OverlappingNSGAII] Starting | "
            f"pop={self.pop_size} | gen={self.max_gen} | "
            f"initial_labels~{self.n_communities} | "
            f"max_memberships={self.max_memberships}"
        )

        # Generational loop — identical structure to baseline
        for gen in range(self.max_gen):

            # Create offspring
            offspring = []
            for _ in range(self.pop_size):
                parent1, parent2 = random.sample(population, 2)

                # Topology-aware crossover
                if random.random() < self.crossover_prob:
                    child = _topology_aware_crossover(
                        parent1, parent2,
                        self.max_memberships,
                        self.neighbors_cache,
                    )
                else:
                    child = {node: set(cids) for node, cids in parent1.items()}

                # Topology-aware additive mutation
                child = _topology_aware_mutation(
                    child,
                    self.mutation_prob,
                    self.max_memberships,
                    self.neighbors_cache,
                    self.max_label_id,
                )

                offspring.append(child)

            # Evaluate offspring
            offspring_fitnesses = [_evaluate_overlapping(self.G, ind) for ind in offspring]

            # Combine + sort + select — identical to baseline
            combined           = population + offspring
            combined_fitnesses = fitnesses  + offspring_fitnesses
            fronts             = self._fast_non_dominated_sort(combined_fitnesses)

            new_population, new_fitnesses = [], []
            for front in fronts:
                if len(new_population) + len(front) <= self.pop_size:
                    new_population  += [combined[i]           for i in front]
                    new_fitnesses   += [combined_fitnesses[i] for i in front]
                else:
                    needed = self.pop_size - len(new_population)
                    new_population  += [combined[i]           for i in front[:needed]]
                    new_fitnesses   += [combined_fitnesses[i] for i in front[:needed]]
                    break

            population, fitnesses = new_population, new_fitnesses

            if (gen + 1) % 10 == 0:
                best_q = max(1 - f1 - f2 for f1, f2 in fitnesses)
                print(f"  gen {gen+1:3d}/{self.max_gen} | best Q = {best_q:.4f}")

        # Select best by Q = 1 - f1 - f2 — identical to baseline
        best_idx       = max(range(len(fitnesses)), key=lambda i: 1 - fitnesses[i][0] - fitnesses[i][1])
        best_partition = population[best_idx]
        best_f1, best_f2 = fitnesses[best_idx]

        result = _partition_to_frozensets(best_partition)
        overlapping_nodes = sum(1 for cids in best_partition.values() if len(cids) > 1)

        print(
            f"[OverlappingNSGAII] Done | "
            f"communities={len(result)} | "
            f"overlapping_nodes={overlapping_nodes} | "
            f"Q={1-best_f1-best_f2:.4f}"
        )
        return result


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_hp_mocd_overlapping(
    G: nx.Graph,
    cfg: dict        = HPMOCD_CONFIG,
    max_memberships: int = 2,
    n_communities: int | None = None,
) -> tuple[list[frozenset], float]:
    """
    Run overlapping HP-MOCD extension on graph G.
    Mirrors run_hp_mocd() signature from hp_mocd_baseline.py.

    Parameters
    ----------
    G               : nx.Graph
    cfg             : dict  — same config as baseline
    max_memberships : int   — max communities per node (default 2, try 3 for DBLP)
    n_communities   : int?  — initial community-label pool size.
                               If None, uses cfg["n_communities"] if present,
                               otherwise falls back to heuristic.

    Returns
    -------
    list[frozenset], float  — partition and runtime in seconds
    """
    t0      = time.perf_counter()
    model   = OverlappingNSGAII(
        G,
        cfg=cfg,
        max_memberships=max_memberships,
        n_communities=n_communities,
    )
    best    = model.run()
    runtime = time.perf_counter() - t0

    print(f"[HP-MOCD Overlapping] communities={len(best)}, runtime={runtime:.2f}s")
    return best, runtime


# ══════════════════════════════════════════════════════════════════════════════
#  QUICK SMOKE-TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    sys.path.insert(0, "..")

    try:
        from data.load_lfr import load_lfr_overlapping, load_lfr_disjoint
        try:
            G, ground_truth = load_lfr_overlapping()
            print(f"LFR (compat path): {G.number_of_nodes()} nodes, "
                  f"{G.number_of_edges()} edges, "
                  f"{len(ground_truth)} ground-truth communities")
        except Exception as e:
            print(f"Overlapping LFR path failed ({e}), falling back to disjoint LFR.")
            G, ground_truth = load_lfr_disjoint()
            print(f"LFR disjoint: {G.number_of_nodes()} nodes, "
                  f"{G.number_of_edges()} edges, "
                  f"{len(ground_truth)} ground-truth communities")
    except ImportError as e:
        print(f"LFR loader import failed ({e}), using Karate Club instead.")
        G = nx.karate_club_graph()
        ground_truth = None

    partition, runtime = run_hp_mocd_overlapping(G, max_memberships=2)

    print(f"\nResult: {len(partition)} communities in {runtime:.2f}s")
    node_counts: Counter = Counter()
    for community in partition:
        for node in community:
            node_counts[node] += 1
    n_overlapping = sum(1 for c in node_counts.values() if c > 1)
    print(f"Overlapping nodes: {n_overlapping} / {G.number_of_nodes()}")
    if ground_truth:
        print(f"Ground-truth communities: {len(ground_truth)}")
