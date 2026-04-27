"""
Utilities to inspect and persist community partitions in a common format.

Both baseline and overlapping methods already return list[frozenset].
This module adds a deterministic node -> communities view for easy comparison.
"""

from __future__ import annotations

import json
from pathlib import Path


def partition_to_node_memberships(partition: list[frozenset]) -> dict[int, list[int]]:
    """
    Convert list[frozenset] communities to node -> sorted community ids.

    Example
    -------
    partition = [frozenset({1, 2, 3}), frozenset({3, 4})]
    returns   = {1: [0], 2: [0], 3: [0, 1], 4: [1]}
    """
    node_to_communities: dict[int, list[int]] = {}

    for cid, community in enumerate(partition):
        for node in community:
            node_to_communities.setdefault(node, []).append(cid)

    for node in node_to_communities:
        node_to_communities[node].sort()

    # Stable ordering makes diffs and debugging easier.
    return dict(sorted(node_to_communities.items(), key=lambda item: item[0]))


def partition_to_serializable(partition: list[frozenset]) -> list[list[int]]:
    """Convert list[frozenset] to JSON-friendly sorted list[list[int]]."""
    return [sorted(list(community)) for community in partition]


def save_partition_report(
    output_path: Path,
    dataset_name: str,
    method_name: str,
    partition: list[frozenset],
) -> dict[int, list[int]]:
    """
    Save a partition report with both views:
      1) community_to_nodes  (for metric pipelines)
      2) node_to_communities (for human inspection)

    Returns node_to_communities so callers can print previews without recomputing.
    """
    node_to_communities = partition_to_node_memberships(partition)

    payload = {
        "dataset": dataset_name,
        "method": method_name,
        "community_to_nodes": partition_to_serializable(partition),
        "node_to_communities": {
            str(node): memberships
            for node, memberships in node_to_communities.items()
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return node_to_communities
