"""
data/load_lfr.py — Generate LFR benchmark graphs.

NetworkX provides a disjoint LFR generator. The overlapping loader keeps the
same project-facing API name for compatibility but currently uses the same
backend generation path.
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
    communities = list({frozenset(G.nodes[v]["community"]) for v in G})
    G_clean = nx.Graph(G)
    nx.set_node_attributes(G_clean, {v: G.nodes[v]["community"] for v in G}, "community")
    print(f"[LFR disjoint] nodes={G_clean.number_of_nodes()}, "
          f"edges={G_clean.number_of_edges()}, "
          f"communities={len(communities)}")
    return G_clean, communities


def load_lfr_overlapping(cfg: dict = LFR_CONFIG) -> tuple[nx.Graph, list[frozenset]]:
    """
    Compatibility loader for the project's overlapping path.

    Note:
    NetworkX's LFR_benchmark_graph does not expose overlap parameters (on/om),
    so this currently generates standard LFR communities.

    Returns
    -------
    G : nx.Graph
    communities : list of frozensets
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

    communities = list({frozenset(G.nodes[v]["community"]) for v in G})
    G_clean = nx.Graph(G)
    nx.set_node_attributes(G_clean, {v: G.nodes[v]["community"] for v in G}, "community")

    print(f"[LFR compatibility] nodes={G_clean.number_of_nodes()}, "
          f"edges={G_clean.number_of_edges()}, "
          f"communities={len(communities)}")
    return G_clean, communities


# ── Quick sanity check ───────────────────────────────────────────────────────
if __name__ == "__main__":
    G_d, cmty_d = load_lfr_disjoint()
    print(f"Disjoint: {len(cmty_d)} communities")

    try:
        G_o, cmty_o = load_lfr_overlapping()
        print(f"Overlapping: {len(cmty_o)} communities")
    except TypeError as e:
        print(f"Overlapping LFR not available: {e}")
