"""
evaluation/metrics.py
─────────────────────
Evaluation metrics for both disjoint and overlapping community detection.

Disjoint metrics  : NMI, AMI, Modularity, F1 (pairwise)
Overlapping metrics: Omega Index, Overlapping NMI (ONMI)

All functions accept communities as list[frozenset[int]].

IMPORTANT — OVERLAPPING COVERS AND NMI
───────────────────────────────────────
Standard sklearn NMI requires a hard (disjoint) partition.  For overlapping
predictions the naive approach of projecting each node to its "last seen"
community produces a broken label array that systematically underestimates NMI.

We handle this in two ways:

  1. nmi()  — used for the table.  For overlapping covers we project each node
     to its BEST-SUPPORTED community (using neighbourhood majority vote), which
     is the same projection used inside the algorithm.  This gives a fair
     comparison: the same projection is used for both eval and optimisation.

  2. onmi() — true overlapping NMI as defined by McDaid et al. (2011).
     Computes H(C|D) and H(D|C) using the cover membership vectors.
     This is the gold-standard metric for overlapping CD evaluation.

MODULARITY FOR OVERLAPPING COVERS
──────────────────────────────────
NetworkX modularity() requires a disjoint partition.  We project using the
support-based hard partition (same as the algorithm) rather than first-seen,
so the reported modularity matches what was actually optimised.
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
    For overlapping covers, each node is assigned to its FIRST community
    only — callers that need overlapping-aware projection should use
    _cover_to_label_array_support_based() instead.
    """
    label = np.full(len(nodes), -1, dtype=int)
    node_idx = {v: i for i, v in enumerate(nodes)}
    for cid, comm in enumerate(communities):
        for v in comm:
            if v in node_idx and label[node_idx[v]] == -1:
                label[node_idx[v]] = cid
    return label


def _cover_to_label_array_support_based(
    communities: list[frozenset],
    nodes: list,
    G: nx.Graph | None = None,
) -> np.ndarray:
    """
    Project an overlapping cover to a label array using best-supported assignment.

    For each node that appears in multiple communities:
      - If G is provided: assign to the community that contains the most
        of the node's neighbours (neighbourhood majority vote).
      - If G is None: assign to the largest community (size proxy).

    This is the same projection used inside the algorithm for f1/f2, so
    NMI and Modularity computed with this projection are consistent with
    what was actually optimised.
    """
    node_idx = {v: i for i, v in enumerate(nodes)}
    label = np.full(len(nodes), -1, dtype=int)

    # Build node → {cid: community_set} map
    node_to_comms: dict = {}
    for cid, comm in enumerate(communities):
        for v in comm:
            node_to_comms.setdefault(v, {})[cid] = comm

    for v, cid_map in node_to_comms.items():
        if v not in node_idx:
            continue
        if len(cid_map) == 1:
            label[node_idx[v]] = next(iter(cid_map))
            continue
        # Multiple communities — pick the best-supported one.
        if G is not None and v in G:
            nbrs = set(G.neighbors(v))
            best_cid = max(
                cid_map.keys(),
                key=lambda cid: len(nbrs & cid_map[cid])
            )
        else:
            best_cid = max(cid_map.keys(), key=lambda cid: len(cid_map[cid]))
        label[node_idx[v]] = best_cid

    return label


def _cover_to_hard_partition(
    communities: list[frozenset],
    G: nx.Graph | None = None,
) -> list[frozenset]:
    """
    Project an overlapping cover to a disjoint hard partition.

    Uses support-based assignment (neighbourhood majority vote) when G is
    provided.  Falls back to first-seen when G is None.

    This is used for modularity computation.  Using support-based projection
    makes the reported Modularity consistent with what the algorithm
    optimised internally via _hard_partition_from_overlapping().
    """
    nodes = list(set().union(*communities)) if communities else []
    node_idx = {v: i for i, v in enumerate(nodes)}
    labels = _cover_to_label_array_support_based(communities, nodes, G=G)

    groups: dict[int, set] = {}
    for v, idx in node_idx.items():
        cid = int(labels[idx])
        if cid >= 0:
            groups.setdefault(cid, set()).add(v)

    return [frozenset(nodes) for _cid, nodes in sorted(groups.items())]


# ── Disjoint metrics ─────────────────────────────────────────────────────────

def _is_overlapping(communities: list[frozenset]) -> bool:
    """Return True if any node appears in more than one community."""
    seen: set = set()
    for comm in communities:
        for v in comm:
            if v in seen:
                return True
            seen.add(v)
    return False


def nmi(
    pred: list[frozenset],
    true: list[frozenset],
    G: nx.Graph | None = None,
) -> float:
    """
    Normalized Mutual Information.

    For overlapping covers, each node is projected to its best-supported
    community before computing NMI.  This avoids the label-collision bug
    where the last-seen community label overwrites earlier ones, which
    systematically underestimates NMI for overlapping predictions.

    For disjoint covers the behaviour is identical to before.
    """
    nodes = list(set().union(*pred, *true))
    pred_overlapping = _is_overlapping(pred)

    if pred_overlapping:
        y_pred = _cover_to_label_array_support_based(pred, nodes, G=G)
    else:
        y_pred = _partition_to_label_array(pred, nodes)

    y_true = _partition_to_label_array(true, nodes)
    mask = (y_pred != -1) & (y_true != -1)
    if mask.sum() == 0:
        return 0.0
    return normalized_mutual_info_score(y_true[mask], y_pred[mask])


def ami(
    pred: list[frozenset],
    true: list[frozenset],
    G: nx.Graph | None = None,
) -> float:
    """Adjusted Mutual Information (overlapping-aware projection)."""
    nodes = list(set().union(*pred, *true))
    pred_overlapping = _is_overlapping(pred)

    if pred_overlapping:
        y_pred = _cover_to_label_array_support_based(pred, nodes, G=G)
    else:
        y_pred = _partition_to_label_array(pred, nodes)

    y_true = _partition_to_label_array(true, nodes)
    mask = (y_pred != -1) & (y_true != -1)
    if mask.sum() == 0:
        return 0.0
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


def onmi(pred: list[frozenset], true: list[frozenset]) -> float:
    """
    Overlapping Normalized Mutual Information (McDaid et al., 2011).

    The standard NMI generalised to overlapping covers.  Each node is
    represented as a binary membership vector over communities, and the
    NMI is computed between the two sets of membership vectors.

    H(C_i) = entropy of community C_i's membership vector across nodes.
    H(C_i | D_j) = conditional entropy of C_i given D_j.

    onmi = 1 - 0.5 * (H(C|D)/H(C) + H(D|C)/H(D))

    Range: [0, 1].  1 = perfect agreement.

    This metric is appropriate when both pred and true are overlapping covers.
    It correctly credits partial overlap matches that Omega and projected NMI
    both miss.
    """
    def _h(p: float) -> float:
        """Binary entropy."""
        if p <= 0 or p >= 1:
            return 0.0
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

    nodes = sorted(set().union(*pred, *true))
    n = len(nodes)
    if n == 0:
        return 1.0
    node_idx = {v: i for i, v in enumerate(nodes)}

    def _membership_matrix(communities: list[frozenset]) -> np.ndarray:
        """Binary matrix: rows = communities, cols = nodes."""
        mat = np.zeros((len(communities), n), dtype=np.float32)
        for cid, comm in enumerate(communities):
            for v in comm:
                if v in node_idx:
                    mat[cid, node_idx[v]] = 1.0
        return mat

    P = _membership_matrix(pred)   # shape (|pred|, n)
    T = _membership_matrix(true)   # shape (|true|, n)

    def _community_entropy(mat: np.ndarray) -> np.ndarray:
        """H(C_i) for each community row."""
        p = mat.mean(axis=1)       # fraction of nodes in each community
        return np.array([_h(pi) for pi in p])

    def _conditional_entropy_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        H(A_i | B_j) for all pairs.  Shape: (|A|, |B|).
        Uses the joint membership distribution over nodes.
        """
        n_a, n_b = A.shape[0], B.shape[0]
        H_A_given_B = np.zeros((n_a, n_b))
        for i in range(n_a):
            a = A[i]            # binary vector length n
            pa = a.mean()
            for j in range(n_b):
                b = B[j]
                pb = b.mean()
                # Joint probabilities
                p11 = np.mean(a * b)
                p10 = pa - p11
                p01 = pb - p11
                p00 = 1 - pa - pb + p11
                joint = [p00, p01, p10, p11]
                # H(A_i, B_j)
                h_joint = -sum(p * np.log2(p) for p in joint if p > 0)
                # H(B_j)
                h_b = _h(pb)
                # H(A_i | B_j) = H(A_i, B_j) - H(B_j)
                H_A_given_B[i, j] = max(0.0, h_joint - h_b)
        return H_A_given_B

    H_P = _community_entropy(P)    # (|pred|,)
    H_T = _community_entropy(T)    # (|true|,)

    # H(P_i | T) = min_j H(P_i | T_j)
    H_P_given_T_matrix = _conditional_entropy_matrix(P, T)
    H_T_given_P_matrix = _conditional_entropy_matrix(T, P)

    # For each community in pred, pick the best-matching true community.
    H_P_given_T = H_P_given_T_matrix.min(axis=1)   # (|pred|,)
    H_T_given_P = H_T_given_P_matrix.min(axis=1)   # (|true|,)

    # Normalise and average.
    # Communities with H=0 (all-in or all-out) contribute 0.
    with np.errstate(divide="ignore", invalid="ignore"):
        norm_P = np.where(H_P > 0, H_P_given_T / H_P, 0.0)
        norm_T = np.where(H_T > 0, H_T_given_P / H_T, 0.0)

    if len(norm_P) == 0 or len(norm_T) == 0:
        return 0.0

    return float(1.0 - 0.5 * (norm_P.mean() + norm_T.mean()))


# ── Convenience: run all metrics at once ─────────────────────────────────────

def evaluate_disjoint(
    G: nx.Graph,
    pred: list[frozenset],
    true: list[frozenset],
) -> dict[str, float]:
    return {
        "NMI":        nmi(pred, true, G=G),
        "AMI":        ami(pred, true, G=G),
        "Modularity": modularity(G, pred),
        "F1":         pairwise_f1(pred, true),
    }


def evaluate_overlapping(
    G: nx.Graph,
    pred: list[frozenset],
    true: list[frozenset],
) -> dict[str, float]:
    pred_hard = _cover_to_hard_partition(pred, G=G)   # support-based projection
    return {
        "NMI":        nmi(pred, true, G=G),
        "AMI":        ami(pred, true, G=G),
        "Modularity": modularity(G, pred_hard),
        "F1":         pairwise_f1(pred, true),
        "Omega":      omega_index(pred, true),
        "ONMI":       onmi(pred, true),
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