"""
evaluation/metrics.py
─────────────────────
Evaluation metrics for both disjoint and overlapping community detection.

Disjoint metrics  : NMI, AMI, Modularity, F1 (pairwise)
Overlapping metric: Omega Index

All functions accept communities as list[frozenset[int]].
"""

from __future__ import annotations
import itertools
import numpy as np
import networkx as nx
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score
from scipy.optimize import linear_sum_assignment


# ── Helpers ──────────────────────────────────────────────────────────────────

def _partition_to_label_array(
    communities: list[frozenset],
    nodes: list,
) -> np.ndarray:
    """
    Convert a disjoint partition to a label array aligned with `nodes`.
    Nodes not assigned to any community get label -1.
    """
    label = np.full(len(nodes), -1, dtype=int)
    node_idx = {v: i for i, v in enumerate(nodes)}
    for cid, comm in enumerate(communities):
        for v in comm:
            if v in node_idx:
                label[node_idx[v]] = cid
    return label


# ── Disjoint metrics ─────────────────────────────────────────────────────────

def nmi(pred: list[frozenset], true: list[frozenset]) -> float:
    """Normalized Mutual Information (sklearn implementation)."""
    nodes = list(set().union(*pred, *true))
    y_pred = _partition_to_label_array(pred, nodes)
    y_true = _partition_to_label_array(true, nodes)
    mask = (y_pred != -1) & (y_true != -1)
    return normalized_mutual_info_score(y_true[mask], y_pred[mask])


def ami(pred: list[frozenset], true: list[frozenset]) -> float:
    """Adjusted Mutual Information."""
    nodes = list(set().union(*pred, *true))
    y_pred = _partition_to_label_array(pred, nodes)
    y_true = _partition_to_label_array(true, nodes)
    mask = (y_pred != -1) & (y_true != -1)
    return adjusted_mutual_info_score(y_true[mask], y_pred[mask])


def modularity(G: nx.Graph, communities: list[frozenset]) -> float:
    """Standard Newman-Girvan modularity Q."""
    return nx.community.modularity(G, communities)


def pairwise_f1(pred: list[frozenset], true: list[frozenset]) -> float:
    """
    Pairwise F1 score (paper Section 5.1.1).
    Treats each node-pair as a binary classification:
      positive  = both nodes in the same community
      negative  = different communities
    """
    nodes = list(set().union(*pred, *true))
    node_idx = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)

    def pair_set(communities: list[frozenset]) -> set[tuple]:
        pairs = set()
        for c in communities:
            lst = sorted(node_idx[v] for v in c if v in node_idx)
            for i in range(len(lst)):
                for j in range(i + 1, len(lst)):
                    pairs.add((lst[i], lst[j]))
        return pairs

    pred_pairs = pair_set(pred)
    true_pairs = pair_set(true)

    tp = len(pred_pairs & true_pairs)
    fp = len(pred_pairs - true_pairs)
    fn = len(true_pairs - pred_pairs)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ── Overlapping metric ────────────────────────────────────────────────────────

def omega_index(pred: list[frozenset], true: list[frozenset]) -> float:
    """
    Omega Index — the standard metric for overlapping community detection.

    Reference: Collins & Dent (1988); used in Xie et al. (2011) survey.

    The Omega index measures agreement between two covers by counting
    pairs of nodes that co-occur in exactly k communities, for all k.

    Range: [0, 1]  (1 = perfect agreement)
    """
    nodes = list(set().union(*pred, *true))
    node_idx = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)

    def co_occurrence_count(communities: list[frozenset], ni: int) -> np.ndarray:
        """
        co[i, j] = number of communities that contain BOTH node i and node j.
        Returns upper-triangle values as a dict for sparsity.
        """
        count = np.zeros((ni, ni), dtype=np.int32)
        for c in communities:
            members = [node_idx[v] for v in c if v in node_idx]
            for a, b in itertools.combinations(members, 2):
                count[a, b] += 1
                count[b, a] += 1
        return count

    pred_co = co_occurrence_count(pred, n)
    true_co = co_occurrence_count(true, n)

    # Max co-occurrence across both covers
    max_k = max(pred_co.max(), true_co.max(), 1)

    # Observed agreement
    pairs_total = n * (n - 1) / 2
    if pairs_total == 0:
        return 1.0

    # t_k: number of pairs assigned to exactly k communities in BOTH covers
    observed = sum(
        np.sum((pred_co == k) & (true_co == k)) / 2   # upper triangle only
        for k in range(max_k + 1)
    )

    # Expected agreement (by chance)
    # A_k: fraction of pairs in pred with co-count k
    # B_k: fraction of pairs in true with co-count k
    expected = sum(
        (np.sum(pred_co == k) / 2) * (np.sum(true_co == k) / 2) / (pairs_total ** 2)
        * pairs_total
        for k in range(max_k + 1)
    )

    denom = pairs_total - expected
    if denom <= 0:
        return 1.0

    return (observed - expected) / denom


# ── Convenience: run all metrics at once ─────────────────────────────────────

def evaluate_disjoint(
    G: nx.Graph,
    pred: list[frozenset],
    true: list[frozenset],
) -> dict[str, float]:
    return {
        "NMI":        nmi(pred, true),
        "AMI":        ami(pred, true),
        "Modularity": modularity(G, pred),
        "F1":         pairwise_f1(pred, true),
    }


def evaluate_overlapping(
    G: nx.Graph,
    pred: list[frozenset],
    true: list[frozenset],
) -> dict[str, float]:
    return {
        "NMI":        nmi(pred, true),        # approximate for overlapping
        "AMI":        ami(pred, true),
        "Modularity": modularity(G, pred),
        "F1":         pairwise_f1(pred, true),
        "Omega":      omega_index(pred, true), # primary overlapping metric
    }


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Toy example: 6 nodes, 2 communities, node 2 overlaps
    true = [frozenset({0, 1, 2}), frozenset({2, 3, 4, 5})]
    pred = [frozenset({0, 1, 2}), frozenset({2, 3, 4, 5})]   # perfect

    G = nx.karate_club_graph()
    true_k = [frozenset(c) for c in nx.community.greedy_modularity_communities(G)]

    print("=== Disjoint (perfect) ===")
    print(evaluate_disjoint(G, true_k, true_k))

    print("\n=== Overlapping Omega (perfect) ===")
    print(f"Omega = {omega_index(true, pred):.4f}  (expect ~1.0)")

    pred_noisy = [frozenset({0, 1}), frozenset({2, 3, 4, 5})]
    print(f"Omega (noisy) = {omega_index(pred_noisy, true):.4f}")
