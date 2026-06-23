"""
investigation/motifs.py
------------------------
Laundering motif detection inside a k-hop subgraph.

Three patterns are checked:
  - Smurfing   : many sources → single sink (fan-in ≥ 5)
  - Layering   : directed temporal chain of 3–5 hops
  - Circular   : directed cycle of length ≤ 6
"""

import networkx as nx
from typing import Dict

# Maximum hops considered for layering paths
_LAYER_MIN = 3
_LAYER_MAX = 5
# Maximum allowed gap between consecutive hops (seconds)
_LAYERING_GAP_SEC = 3600          # 60 minutes
# Smurfing limits
_SMURF_FAN_IN = 4
_SMURF_MAX_SINKS = 3


def _edge_timestamps(G: nx.DiGraph, u: str, v: str):
    """Safely retrieve the sorted timestamp list for edge (u, v)."""
    data = G.edges[u, v]
    timestamps = data.get("timestamps")
    if timestamps is None:
        t = data.get("timestamp")
        if t is not None:
            timestamps = [t]
        else:
            timestamps = []
    return timestamps


# ---------------------------------------------------------------------------
# Individual detectors
# ---------------------------------------------------------------------------

def detect_smurfing(G: nx.DiGraph) -> bool:
    """
    Return True if a node has >= 4 incoming neighbors
    AND those neighbors connect to <= 3 unique downstream nodes.
    """
    for node in G.nodes():
        in_neighbors = list(G.predecessors(node))
        if len(in_neighbors) < _SMURF_FAN_IN:
            continue

        # Collect unique out-neighbours of the in-neighbours
        unique_sinks = set()
        for src in in_neighbors:
            for dst in G.successors(src):
                unique_sinks.add(dst)
                
        if len(unique_sinks) <= _SMURF_MAX_SINKS:
            return True

    return False


def detect_layering(G: nx.DiGraph) -> bool:
    """
    Return True if a directed temporal chain of length 3–5 exists where:
      - edges occur in monotonically increasing timestamp order
      - each consecutive hop gap is ≤ 60 minutes
    """
    nodes = list(G.nodes())
    state_checks = 0
    MAX_STATE_CHECKS = 5000  # Guard against path explosion in dense networks

    for source in nodes:
        # DFS-style path exploration (bounded depth)
        stack = [(source, [source])]
        while stack:
            state_checks += 1
            if state_checks > MAX_STATE_CHECKS:
                return False  # Early exit to prevent lockups

            current, path = stack.pop()

            path_len = len(path) - 1  # number of edges so far

            if _LAYER_MIN <= path_len <= _LAYER_MAX:
                # Verify temporal ordering along the collected path
                valid = True
                last_ts = None
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    ts_list = _edge_timestamps(G, u, v)
                    if not ts_list:
                        valid = False
                        break
                    ts = ts_list[0]  # earliest transaction on this edge
                    if last_ts is not None:
                        try:
                            gap = (ts - last_ts).total_seconds()
                        except TypeError:
                            # timestamps may be strings; fall back to string compare
                            gap = 0
                        if gap < 0 or gap > _LAYERING_GAP_SEC:
                            valid = False
                            break
                    last_ts = ts
                if valid:
                    return True

            if path_len < _LAYER_MAX:
                for neighbor in G.successors(current):
                    if neighbor not in path:  # prevent revisits
                        stack.append((neighbor, path + [neighbor]))

    return False


def detect_circular(G: nx.DiGraph) -> bool:
    """
    Return True if any directed cycle of length ≤ 6 exists in G.
    Uses networkx.simple_cycles (generator, stops early on first match).
    """
    for cycle in nx.simple_cycles(G, length_bound=6):
        if len(cycle) <= 6:
            return True
    return False


# ---------------------------------------------------------------------------
# Combined entry-point
# ---------------------------------------------------------------------------

def detect_motifs(G_sub: nx.DiGraph) -> Dict[str, bool]:
    """
    Run all three motif detectors on a subgraph.

    Returns
    -------
    {"smurfing": bool, "layering": bool, "circular": bool}
    """
    return {
        "smurfing": detect_smurfing(G_sub),
        "layering": detect_layering(G_sub),
        "circular": detect_circular(G_sub),
    }
