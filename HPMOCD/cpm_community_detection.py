"""
cpm_community_detection.py
══════════════════════════
Two versions of the Clique Percolation Method (CPM) for overlapping
community detection.

    VERSION 1 — run_cpm_original()
    ────────────────────────────────
    The classic CPM as described in Palla et al. (2005).
    Uses NetworkX's built-in k_clique_communities().
    Nodes not part of any k-clique are LEFT UNASSIGNED (the NCN problem).
    This is the "honest" original — included so you can show exactly what
    the NCN problem looks like in practice.

    VERSION 2 — run_cpm_ncn_fixed()
    ────────────────────────────────
    Same CPM core, but unassigned nodes are recovered by assigning each
    one to the community it shares the most edges with.
    If a node has no edges at all (isolated), it gets its own singleton
    community so that every node is covered.
    This is the "fair" version for metric comparison, since NMI/Omega
    require complete node coverage.

BOTH VERSIONS
    • Return list[frozenset]  — identical format to your baseline and
      overlapping extension, so metrics.py works unchanged.
    • Accept the same G: nx.Graph input.
    • Auto-tune k (tries k=3,4,5 and picks the one with best modularity)
      so you don't have to manually set it per dataset.

WHY CPM FOR COMPARISON
    CPM is the classical standard for overlapping community detection.
    Every overlapping CD paper compares against it. It represents the
    non-evolutionary, pure-structure approach to overlap — finding
    communities as unions of overlapping cliques rather than by
    optimising any objective function.

DISADVANTAGES OF CPM (relevant for your report)
    1. NCN problem     — nodes not in any k-clique are unassigned
    2. k is manual     — results change dramatically with k=3 vs k=4
    3. Not optimisation-based — no objective function, just enumeration
    4. Slow on dense graphs — finding all cliques is NP-hard
    5. Misses large sparse communities that have no dense clique cores

HOW TO USE
──────────
    from cpm_community_detection import run_cpm_original, run_cpm_ncn_fixed

    # Version 1 — original, may have unassigned nodes
    partition, runtime, k_used = run_cpm_original(G)

    # Version 2 — NCN fixed, all nodes assigned
    partition, runtime, k_used = run_cpm_ncn_fixed(G)

    # Both return list[frozenset] — plug directly into metrics.py
"""

import time
from collections import Counter

import networkx as nx


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — K SELECTION
# ══════════════════════════════════════════════════════════════════════════════

def _select_best_k(
    G: nx.Graph,
    k_values: list[int] = [3, 4, 5],
) -> tuple[int, list[frozenset]]:
    """
    Try each value of k and return the one that gives the best modularity.

    k is CPM's only parameter — the minimum clique size required for two
    cliques to be considered part of the same community (they must share
    k-1 nodes). k=3 is most permissive (finds more, smaller communities),
    k=5 is strictest (finds fewer, denser communities).

    We auto-select because:
      - There is no principled way to choose k for an unknown graph
      - Different datasets need different k (LFR vs DBLP)
      - This makes the comparison fair — CPM gets its best shot

    If a k produces zero communities (graph too sparse for that clique
    size), we skip it and try the next one.

    Returns
    -------
    best_k         : int
    best_partition : list[frozenset]  (may have unassigned nodes)
    """
    best_k         = k_values[0]
    best_partition = []
    best_modularity = -1.0

    for k in k_values:
        try:
            communities = list(nx.community.k_clique_communities(G, k))
            communities = [frozenset(c) for c in communities]

            if len(communities) == 0:
                # No communities found at this k — skip
                continue

            # Compute modularity on the communities found
            # (only communities that cover at least some nodes)
            try:
                mod = nx.community.modularity(G, communities)
            except Exception:
                mod = -1.0

            print(f"  [CPM k={k}] communities={len(communities)}, "
                  f"modularity={mod:.4f}")

            if mod > best_modularity:
                best_modularity = mod
                best_k          = k
                best_partition  = communities

        except Exception as e:
            print(f"  [CPM k={k}] failed: {e}")
            continue

    return best_k, best_partition


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — NCN RECOVERY
# ══════════════════════════════════════════════════════════════════════════════

def _recover_unassigned_nodes(
    G: nx.Graph,
    communities: list[frozenset],
) -> list[frozenset]:
    """
    Assign every unassigned node to its most-connected community.

    This is the NCN (Non-Classified Node) fix.

    For each unassigned node v:
      1. Count how many edges v has to each existing community
      2. Assign v to the community it shares the most edges with
      3. If v has NO edges to any community (isolated or all neighbours
         also unassigned), give v its own singleton community

    Note: we add nodes to communities by building new frozensets.
    The original communities are not modified.

    Why this is the right fix
    ─────────────────────────
    The NCN problem makes CPM incomparable with other methods on metrics
    like NMI and Omega, because those metrics require every node to be
    assigned. This fix uses the same "neighbourhood majority" principle
    that HP-MOCD's mutation uses — a node belongs where its neighbours are.
    """
    # Find which nodes are already assigned
    assigned_nodes: set = set()
    for community in communities:
        assigned_nodes.update(community)

    unassigned_nodes = set(G.nodes()) - assigned_nodes

    if not unassigned_nodes:
        # Nothing to fix
        return communities

    print(f"  [CPM NCN] Recovering {len(unassigned_nodes)} unassigned nodes ...")

    # Work with mutable sets, convert back to frozensets at the end
    mutable_communities: list[set] = [set(c) for c in communities]

    for node in unassigned_nodes:
        # Count edges from this node to each community
        edge_counts: Counter = Counter()
        for neighbour in G.neighbors(node):
            for cid, community in enumerate(mutable_communities):
                if neighbour in community:
                    edge_counts[cid] += 1

        if edge_counts:
            # Assign to the community with most connections
            best_community_idx = edge_counts.most_common(1)[0][0]
            mutable_communities[best_community_idx].add(node)
        else:
            # Isolated node or all neighbours also unassigned
            # Give it a singleton community so it's not lost
            mutable_communities.append({node})

    return [frozenset(c) for c in mutable_communities if c]


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — VERSION 1: ORIGINAL CPM (with NCN problem)
# ══════════════════════════════════════════════════════════════════════════════

def run_cpm_original(
    G: nx.Graph,
    k_values: list[int] = [3, 4, 5],
) -> tuple[list[frozenset], float, int]:
    """
    Run original CPM — the classic Palla et al. (2005) algorithm.

    Nodes not part of any k-clique are LEFT UNASSIGNED.
    This intentionally preserves the NCN problem so you can:
      (a) show it exists in your results
      (b) compare fairly with Version 2 to show the impact of fixing it

    Parameters
    ----------
    G        : nx.Graph — input graph
    k_values : list of k values to try (best modularity is selected)

    Returns
    -------
    partition : list[frozenset]  — overlapping communities
                                   WARNING: may not cover all nodes
    runtime   : float            — wall-clock seconds
    k_used    : int              — the k value that was selected
    """
    t0 = time.perf_counter()

    print(f"[CPM Original] Trying k values: {k_values} ...")
    best_k, partition = _select_best_k(G, k_values)

    runtime = time.perf_counter() - t0

    # Count coverage statistics
    assigned_nodes = set()
    for community in partition:
        assigned_nodes.update(community)
    unassigned_count = G.number_of_nodes() - len(assigned_nodes)

    # Count overlapping nodes (in more than one community)
    node_counts: Counter = Counter()
    for community in partition:
        for node in community:
            node_counts[node] += 1
    overlapping_count = sum(1 for c in node_counts.values() if c > 1)

    print(f"[CPM Original] k={best_k} | "
          f"communities={len(partition)} | "
          f"overlapping_nodes={overlapping_count} | "
          f"UNASSIGNED={unassigned_count} | "
          f"runtime={runtime:.2f}s")

    return partition, runtime, best_k


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — VERSION 2: CPM WITH NCN FIX
# ══════════════════════════════════════════════════════════════════════════════

def run_cpm_ncn_fixed(
    G: nx.Graph,
    k_values: list[int] = [3, 4, 5],
) -> tuple[list[frozenset], float, int]:
    """
    Run CPM with NCN (Non-Classified Node) recovery.

    Same core as Version 1, but unassigned nodes are recovered by
    assigning them to their most-connected community. Every node
    is guaranteed to appear in at least one community.

    This version is suitable for fair metric comparison with your
    overlapping extension and HP-MOCD baseline, since NMI and Omega
    require complete node coverage.

    Parameters
    ----------
    G        : nx.Graph — input graph
    k_values : list of k values to try (best modularity is selected)

    Returns
    -------
    partition : list[frozenset]  — overlapping communities (ALL nodes covered)
    runtime   : float            — wall-clock seconds
    k_used    : int              — the k value that was selected
    """
    t0 = time.perf_counter()

    print(f"[CPM NCN-Fixed] Trying k values: {k_values} ...")
    best_k, partition = _select_best_k(G, k_values)

    # Recover unassigned nodes
    partition = _recover_unassigned_nodes(G, partition)

    runtime = time.perf_counter() - t0

    # Count coverage statistics
    node_counts: Counter = Counter()
    for community in partition:
        for node in community:
            node_counts[node] += 1
    overlapping_count = sum(1 for c in node_counts.values() if c > 1)

    # Verify full coverage
    all_nodes_covered = set(node_counts.keys()) == set(G.nodes())

    print(f"[CPM NCN-Fixed] k={best_k} | "
          f"communities={len(partition)} | "
          f"overlapping_nodes={overlapping_count} | "
          f"all_nodes_covered={all_nodes_covered} | "
          f"runtime={runtime:.2f}s")

    return partition, runtime, best_k


# ══════════════════════════════════════════════════════════════════════════════
#  QUICK SMOKE-TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")

    try:
        from data.load_lfr import load_lfr_overlapping, load_lfr_disjoint
        try:
            G, ground_truth = load_lfr_overlapping()
            print(f"LFR overlapping: {G.number_of_nodes()} nodes, "
                  f"{G.number_of_edges()} edges, "
                  f"{len(ground_truth)} ground-truth communities")
        except Exception as e:
            print(f"Overlapping LFR not available ({e}), using disjoint.")
            G, ground_truth = load_lfr_disjoint()
    except Exception:
        print("Using Karate Club graph.")
        G = nx.karate_club_graph()
        ground_truth = None

    print("\n── Version 1: Original CPM ──")
    p1, rt1, k1 = run_cpm_original(G)
    assigned1 = set().union(*p1) if p1 else set()
    print(f"Covered {len(assigned1)}/{G.number_of_nodes()} nodes")

    print("\n── Version 2: CPM with NCN fix ──")
    p2, rt2, k2 = run_cpm_ncn_fixed(G)
    assigned2 = set().union(*p2) if p2 else set()
    print(f"Covered {len(assigned2)}/{G.number_of_nodes()} nodes")
