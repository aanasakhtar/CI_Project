"""
slpa.py
-------
Speaker-Listener Label Propagation Algorithm (SLPA) for overlapping
community detection.

Reference:
    Xie, J., Szymanski, B. K., & Liu, X. (2011).
    SLPA: Uncovering Overlapping Communities in Social Networks via
    a Speaker-listener Interaction Dynamic Process.
    IEEE ICDM Workshops. https://arxiv.org/abs/1109.5720

Algorithm summary
-----------------
Each node maintains a memory of labels (community ids) it has received.
Over T rounds:
  1. SPEAK  — each node selects a label from its memory proportional to
              frequency and broadcasts it to its neighbours.
  2. LISTEN — each node receives labels from all its speakers, selects
              the most popular one (ties broken randomly), and stores it.
After T rounds, each node's memory is post-processed with a threshold r:
labels that appear less than r * T times are discarded.
The remaining labels define the node's community memberships.

Why SLPA is a good fit for this project
-----------------------------------------
• O(T·m) time — linear in edges per iteration, very fast on both
  LFR (1 000 nodes) and the 10 k-node DBLP subsample.
• Natural overlapping output — no explicit overlap forcing.
  Overlap emerges from the label memory and threshold post-processing.
• Tunable overlap density — lower r → more overlapping nodes.
• Well-documented benchmark results on LFR and DBLP (Xie et al. 2011).

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


def _slpa_core(
    G: nx.Graph,
    T: int,
    r: float,
    seed: int,
) -> list[frozenset]:
    """
    Run SLPA and return overlapping communities.

    Parameters
    ----------
    G    : undirected graph
    T    : number of propagation rounds  (default 20 is standard)
    r    : post-processing threshold in (0, 1)
           labels kept only if frequency > r * T
    seed : random seed for reproducibility

    Returns
    -------
    list[frozenset]  — overlapping community cover (all nodes assigned)
    """
    rng = random.Random(seed)
    nodes = list(G.nodes())

    # Step 1 — initialise: every node gets its own unique label
    # memory[v] = Counter of label -> count
    memory: dict = {v: Counter({v: 1}) for v in nodes}

    # Step 2 — T propagation rounds
    for _t in range(T):
        # Randomise node processing order each round
        order = list(nodes)
        rng.shuffle(order)

        for listener in order:
            neighbours = list(G.neighbors(listener))
            if not neighbours:
                continue  # isolated node keeps its own label

            # Each neighbour speaks: pick a label from its memory
            # proportionally to frequency
            received: list = []
            for speaker in neighbours:
                mem = memory[speaker]
                total = sum(mem.values())
                # weighted random selection
                labels = list(mem.keys())
                weights = [mem[l] / total for l in labels]
                chosen = rng.choices(labels, weights=weights, k=1)[0]
                received.append(chosen)

            # Listener adopts the most popular label (random tie-break)
            counts = Counter(received)
            max_count = max(counts.values())
            candidates = [l for l, c in counts.items() if c == max_count]
            adopted = rng.choice(candidates)
            memory[listener][adopted] = memory[listener].get(adopted, 0) + 1

    # Step 3 — post-processing: threshold r
    # Keep label l for node v if memory[v][l] / sum(memory[v].values()) > r
    node_labels: dict = {}
    for v in nodes:
        mem = memory[v]
        total = sum(mem.values())
        kept = {l for l, c in mem.items() if c / total > r}
        if not kept:
            # Always keep at least the most frequent label
            kept = {max(mem, key=lambda l: mem[l])}
        node_labels[v] = kept

    # Step 4 — build community cover: group nodes by shared labels
    label_to_nodes: dict = {}
    for v, labels in node_labels.items():
        for l in labels:
            label_to_nodes.setdefault(l, set()).add(v)

    # Filter trivially small communities (singleton labels are noise)
    communities = [
        frozenset(members)
        for members in label_to_nodes.values()
        if len(members) >= 2
    ]

    # Ensure every node is assigned to at least one community
    assigned = set().union(*communities) if communities else set()
    unassigned = set(nodes) - assigned
    if unassigned:
        # Assign each unassigned node to the community it has most edges with
        for v in unassigned:
            best_cid = None
            best_count = -1
            nbrs = set(G.neighbors(v))
            for cid, comm in enumerate(communities):
                count = len(nbrs & comm)
                if count > best_count:
                    best_count = count
                    best_cid = cid
            if best_cid is not None:
                communities[best_cid] = frozenset(communities[best_cid] | {v})
            else:
                communities.append(frozenset({v}))

    return communities


# ── Public runner ─────────────────────────────────────────────────────────────

def run_slpa(
    G: nx.Graph,
    T: int = 20,
    r: float = 0.1,
    seed: int | None = None,
    cfg: dict = HPMOCD_CONFIG,
) -> tuple[list[frozenset], float]:
    """
    Run SLPA on graph G.

    Parameters
    ----------
    G    : nx.Graph — undirected, unweighted input graph
    T    : propagation rounds (default 20; paper recommends 20–100)
    r    : post-processing threshold (default 0.1; lower → more overlaps)
    seed : random seed (falls back to cfg["seed"] or 42)
    cfg  : config dict (used only for seed fallback)

    Returns
    -------
    partition : list[frozenset]  — overlapping community cover
    runtime   : float            — wall-clock seconds
    """
    if seed is None:
        seed = cfg.get("seed", 42)

    t0 = time.perf_counter()
    partition = _slpa_core(G, T=T, r=r, seed=seed)
    runtime = time.perf_counter() - t0

    # Diagnostics
    node_counts: Counter = Counter()
    for community in partition:
        for node in community:
            node_counts[node] += 1
    overlapping = sum(1 for c in node_counts.values() if c > 1)
    assigned = len(node_counts)

    print(
        f"[SLPA] T={T} r={r} | "
        f"communities={len(partition)} | "
        f"overlapping_nodes={overlapping}/{G.number_of_nodes()} | "
        f"assigned={assigned}/{G.number_of_nodes()} | "
        f"runtime={runtime:.2f}s"
    )
    return partition, runtime


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.path.insert(0, "..")
    try:
        from data.load_lfr import load_lfr_overlapping
        G, gt = load_lfr_overlapping()
        print(f"LFR overlapping: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    except Exception as e:
        print(f"LFR not available ({e}), using karate club.")
        G = nx.karate_club_graph()
        gt = None

    partition, rt = run_slpa(G)
    print(f"\nDetected {len(partition)} communities in {rt:.2f}s")
    if gt:
        from evaluation.metrics import evaluate_overlapping
        scores = evaluate_overlapping(G, partition, gt)
        print("Scores:", scores)