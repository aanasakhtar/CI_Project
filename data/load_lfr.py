"""
data/load_lfr.py — Generate LFR benchmark graphs (disjoint & overlapping).

The overlapping variant uses the om / oc parameters supported by
networkx's LFR generator (networkx >= 3.0).
"""

import networkx as nx
from networkx.generators.community import LFR_benchmark_graph
from config import LFR_CONFIG


def load_lfr_disjoint(cfg: dict = LFR_CONFIG) -> tuple[nx.Graph, list[frozenset]]:
    """
    Generate a disjoint LFR graph.

    Returns
    -------
    G : nx.Graph
    communities : list of frozensets  (ground-truth partition)
    """
    G = LFR_benchmark_graph(
        n=cfg["n"],
        tau1=cfg["tau1"],
        tau2=cfg["tau2"],
        mu=cfg["mu"],
        average_degree=cfg["average_degree"],
        max_degree=cfg["max_degree"],
        min_community=cfg["min_community"],
        max_community=cfg["max_community"],
        seed=cfg["seed"],
    )
    # Ground-truth stored as node attribute "community"
    communities = list({frozenset(G.nodes[v]["community"]) for v in G})
    # Strip attributes for clean graph
    G_clean = nx.Graph(G)
    nx.set_node_attributes(G_clean, {v: G.nodes[v]["community"] for v in G}, "community")
    print(f"[LFR disjoint] nodes={G_clean.number_of_nodes()}, "
          f"edges={G_clean.number_of_edges()}, "
          f"communities={len(communities)}")
    return G_clean, communities


def load_lfr_overlapping(cfg: dict = LFR_CONFIG) -> tuple[nx.Graph, list[frozenset]]:
    """
    Generate an overlapping LFR graph.

    Uses the 'om' (overlap membership) and 'on' (overlap nodes) parameters.
    These are available in networkx >= 3.0.

    Returns
    -------
    G : nx.Graph
    communities : list of frozensets  (ground-truth — nodes may appear in multiple sets)
    """
    G = LFR_benchmark_graph(
        n=cfg["n"],
        tau1=cfg["tau1"],
        tau2=cfg["tau2"],
        mu=cfg["mu"],
        average_degree=cfg["average_degree"],
        max_degree=cfg["max_degree"],
        min_community=cfg["min_community"],
        max_community=cfg["max_community"],
        on=cfg["overlap_n"],
        om=cfg["overlap_membership"],
        seed=cfg["seed"],
    )
    # Each node's "community" attribute is a SET of community ids
    # Build list of community frozensets
    cmty_map: dict[int, set] = {}
    for v in G.nodes():
        for c in G.nodes[v]["community"]:
            cmty_map.setdefault(c, set()).add(v)
    communities = [frozenset(members) for members in cmty_map.values()]

    overlapping_nodes = [v for v in G if len(G.nodes[v]["community"]) > 1]
    print(f"[LFR overlapping] nodes={G.number_of_nodes()}, "
          f"edges={G.number_of_edges()}, "
          f"communities={len(communities)}, "
          f"overlapping_nodes={len(overlapping_nodes)}")
    return G, communities


# ── Quick sanity check ───────────────────────────────────────────────────────
if __name__ == "__main__":
    G_d, cmty_d = load_lfr_disjoint()
    G_o, cmty_o = load_lfr_overlapping()
