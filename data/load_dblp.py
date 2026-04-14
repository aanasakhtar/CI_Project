"""
data/load_dblp.py — Download and load the SNAP DBLP co-authorship network.

Source: https://snap.stanford.edu/data/com-DBLP.html
  - com-DBLP.ungraph.txt.gz   : edge list
  - com-DBLP.all.cmty.txt.gz  : ground-truth overlapping communities
"""

import gzip
import os
import random
import urllib.request
from pathlib import Path

import networkx as nx
from config import DBLP_CONFIG


def _download(url: str, dest: Path) -> None:
    if not dest.exists():
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"  Saved to {dest}")
    else:
        print(f"  Already cached: {dest}")


def load_dblp(cfg: dict = DBLP_CONFIG) -> tuple[nx.Graph, list[frozenset]]:
    """
    Download (once) and load the DBLP graph + ground-truth communities.

    Parameters
    ----------
    cfg : dict  (from config.DBLP_CONFIG)

    Returns
    -------
    G          : nx.Graph  (possibly subsampled)
    communities: list of frozensets  (overlapping ground-truth)
    """
    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    graph_gz  = save_dir / "com-DBLP.ungraph.txt.gz"
    cmty_gz   = save_dir / "com-DBLP.all.cmty.txt.gz"

    _download(cfg["url_graph"], graph_gz)
    _download(cfg["url_cmty"],  cmty_gz)

    # ── Load edges ────────────────────────────────────────────────────────────
    print("Parsing edge list ...")
    G = nx.Graph()
    with gzip.open(graph_gz, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            u, v = map(int, line.split())
            G.add_edge(u, v)
    print(f"  Full graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # ── Optional subsampling (BFS from random seed) ───────────────────────────
    subsample = cfg.get("subsample_nodes")
    if subsample and G.number_of_nodes() > subsample:
        rng = random.Random(cfg["seed"])
        start = rng.choice(list(G.nodes()))
        bfs_nodes = list(nx.bfs_tree(G, start).nodes())[:subsample]
        G = G.subgraph(bfs_nodes).copy()
        print(f"  Subsampled: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    node_set = set(G.nodes())

    # ── Load ground-truth communities (filter to subgraph) ───────────────────
    print("Parsing community file ...")
    communities: list[frozenset] = []
    with gzip.open(cmty_gz, "rt") as f:
        for line in f:
            members = frozenset(int(x) for x in line.split()) & node_set
            if len(members) >= 3:          # skip tiny communities
                communities.append(members)

    overlapping = sum(1 for v in node_set
                      if sum(v in c for c in communities) > 1)
    print(f"  Communities (in subgraph): {len(communities)}")
    print(f"  Overlapping nodes: {overlapping} "
          f"({100*overlapping/len(node_set):.1f}%)")

    return G, communities


# ── Quick sanity check ───────────────────────────────────────────────────────
if __name__ == "__main__":
    G, cmty = load_dblp()
    print(f"\nReady: G has {G.number_of_nodes()} nodes, "
          f"{len(cmty)} communities.")
