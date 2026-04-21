"""
detection/metrics.py
--------------------
Compute structural and temporal metrics per node.

Metrics returned per node:
    fanin           - Unique incoming neighbours
    fanout          - Unique outgoing neighbours
    weighted_degree - Total transaction count (in + out)
    burst_score     - Max txs in any 1-h sliding window / avg hourly rate
    chain_score     - Approx 2-hop chain participation score (normalised)
    cycle_score     - 1 if node appears in any simple directed cycle length ≤ 6

Design constraints:
    - No O(N²) nested loops over all node pairs.
    - Timestamps sorted once per node via a pre-built lookup.
    - chain_score uses 2-hop neighbour inspection, not exhaustive DFS.
    - cycle_score limits search to high-activity nodes and caps cycles checked.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from datetime import timedelta
from typing import Any

import networkx as nx
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EPSILON          = 1e-9          # avoid division by zero
BURST_WINDOW_SEC = 3600          # 1-hour sliding window (seconds)
CHAIN_WINDOW_SEC = 7200          # 2-hour forward window for chain detection
CYCLE_MAX_LEN    = 6             # maximum cycle length to check
CYCLE_MIN_TX     = 5             # only check nodes with total_tx_count > this
CYCLE_MAX_PER_NODE = 5           # max cycles to count per node before stopping


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_metrics(
    G: nx.DiGraph,
    node_stats: dict[str, dict[str, Any]],
) -> dict[str, dict[str, float | int]]:
    """
    Compute all structural and temporal metrics for every node in G.

    Parameters
    ----------
    G : nx.DiGraph
    node_stats : dict
        Output of graph_builder.build_graph – per-node transaction counts/amounts.

    Returns
    -------
    node_metrics : dict
        node_id -> {fanin, fanout, weighted_degree, burst_score,
                    chain_score, cycle_score}
    """
    # Pre-build per-node timestamp list (all edges in + out)
    ts_lookup = _build_timestamp_lookup(G)

    # Pre-compute cycle membership for eligible nodes
    cycle_members = _find_cycle_members(G, node_stats)

    node_metrics: dict[str, dict[str, float | int]] = {}

    for node in G.nodes:
        stats = node_stats.get(node, {})
        total_tx = stats.get("total_tx_count", 0)

        fanin  = G.in_degree(node)
        fanout = G.out_degree(node)

        node_metrics[node] = {
            "fanin":          fanin,
            "fanout":         fanout,
            "weighted_degree": total_tx,
            "burst_score":    _burst_score(ts_lookup.get(node, []), total_tx),
            "chain_score":    _chain_score(G, node),
            "cycle_score":    1 if node in cycle_members else 0,
        }

    return node_metrics


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _build_timestamp_lookup(G: nx.DiGraph) -> dict[str, list]:
    """
    Build a dict: node -> sorted list of all timestamps (incoming + outgoing).
    Single pass over edges – avoids repeated iteration later.
    """
    ts_map: dict[str, list] = defaultdict(list)

    for src, dst, data in G.edges(data=True):
        timestamps = data.get("timestamps", [])
        ts_map[src].extend(timestamps)
        ts_map[dst].extend(timestamps)

    # Sort once per node
    for node in ts_map:
        ts_map[node].sort()

    return dict(ts_map)


def _burst_score(timestamps: list, total_tx: int) -> float:
    """
    Z-scored hourly burst detection.
    Bins timestamps into 1-hour windows covering the active period,
    then computes the z-score of the maximum bin.
    """
    if not timestamps or total_tx <= 1:
        return 0.0

    # Calculate hour bucket for each transaction relative to first
    hours = [(t - timestamps[0]).total_seconds() // 3600 for t in timestamps]
    max_hour = int(hours[-1])
    
    if max_hour == 0:
        return 0.0
        
    # Bin transactions into hourly buckets (including intermediate empty hours)
    bucket_counts = np.zeros(max_hour + 1)
    for h in hours:
        bucket_counts[int(h)] += 1
        
    std_dev = np.std(bucket_counts)
    if std_dev < EPSILON:
        return 0.0
        
    z_score = (np.max(bucket_counts) - np.mean(bucket_counts)) / std_dev
    return float(z_score)


def _chain_score(G: nx.DiGraph, node: str) -> float:
    """
    Approximate 3-hop compressed chain participation score.

    For each outgoing edge  node -> A:
        for each outgoing edge A -> B (within window):
            for each outgoing B -> C (within window):
                increment counter

    Normalised by node's out-degree to avoid hub bias.
    """
    out_edges = list(G.out_edges(node, data=True))
    if not out_edges:
        return 0.0

    score = 0
    for _, neighbor_a, edge_data_a in out_edges:
        ts_a = edge_data_a.get("timestamps", [])
        if not ts_a:
            continue
        latest_ts_a = ts_a[-1]
        deadline_a = latest_ts_a + timedelta(seconds=CHAIN_WINDOW_SEC)

        # Check all A -> B edges
        for _, neighbor_b, edge_data_b in G.out_edges(neighbor_a, data=True):
            if neighbor_b == node:
                continue
            ts_b = edge_data_b.get("timestamps", [])
            valid_b = [t for t in ts_b if latest_ts_a <= t <= deadline_a]
            if not valid_b:
                continue
                
            earliest_ts_b = valid_b[0]
            deadline_b = earliest_ts_b + timedelta(seconds=CHAIN_WINDOW_SEC)
            
            # Check all B -> C edges (3rd hop)
            for _, neighbor_c, edge_data_c in G.out_edges(neighbor_b, data=True):
                if neighbor_c in (node, neighbor_a):
                    continue
                ts_c = edge_data_c.get("timestamps", [])
                valid_c = [t for t in ts_c if earliest_ts_b <= t <= deadline_b]
                if valid_c:
                    score += 1

    return score / (len(out_edges) + EPSILON)


def _find_cycle_members(
    G: nx.DiGraph,
    node_stats: dict[str, dict[str, Any]],
) -> set[str]:
    """
    Find nodes that participate in any simple directed cycle of length ≤ 6.

    Uses a bounded DFS per node (not nx.simple_cycles which is exponential):
        - For each eligible node, do a DFS up to depth CYCLE_MAX_LEN.
        - If we reach back to the start node → cycle found, mark and stop DFS
          for that node.
        - Early-exit per node: once one cycle is confirmed, move on.

    Constraints:
        - Only checks nodes with total_tx_count > CYCLE_MIN_TX.
        - DFS depth limited to CYCLE_MAX_LEN (6 hops).
        - CYCLE_MAX_PER_NODE cycles tracked per node before stopping DFS.
        - Runs in O(V * branching_factor^CYCLE_MAX_LEN) which is tractable
          for sparse graphs with small max_len.
    """
    cycle_members: set[str] = set()

    for node, stats in node_stats.items():
        if stats.get("total_tx_count", 0) <= CYCLE_MIN_TX:
            continue
        if node in cycle_members:
            continue  # Already confirmed — skip

        # Bounded DFS: start from `node`, try to return to it within depth 6
        if _has_short_cycle(G, node):
            cycle_members.add(node)

    return cycle_members


def _has_short_cycle(G: nx.DiGraph, start: str) -> bool:
    """
    Return True if there is a simple directed cycle of length ≤ CYCLE_MAX_LEN
    reachable from `start` that returns to `start`.

    Iterative DFS with explicit stack to avoid Python recursion limits.
    Stack entries: (current_node, depth, visited_set_for_this_path)
    """
    # Stack: (current_node, depth)
    # visited_on_path: set of nodes on current path (for simplicity check)
    stack = [(start, 0, {start})]

    while stack:
        current, depth, path = stack.pop()

        if depth >= CYCLE_MAX_LEN:
            continue

        for successor in G.successors(current):
            if successor == start and depth >= 1:
                # Found a cycle back to start
                return True
            if successor not in path and depth + 1 < CYCLE_MAX_LEN:
                stack.append((successor, depth + 1, path | {successor}))

    return False

