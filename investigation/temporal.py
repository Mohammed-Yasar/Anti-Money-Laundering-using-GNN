"""
investigation/temporal.py
--------------------------
Temporal anomaly analysis for AML investigation.

Two signals are computed:
  - burst       : ≥ 3 transactions within any 1-hour sliding window
  - rapid_chain : any two consecutive edge timestamps ≤ 10 minutes apart
"""

from typing import Dict

_BURST_WINDOW_SEC = 3600     # 1 hour
_BURST_MIN_TXS    = 5
_RAPID_GAP_SEC    = 300      # 5 minutes
_MIN_CHAIN_LEN    = 3


def _all_timestamps(G_sub) -> list:
    """
    Collect and sort all timestamps across every edge in the subgraph.
    Edges without a 'timestamps' attribute are silently skipped.
    """
    all_ts = []
    for u, v, data in G_sub.edges(data=True):
        ts_list = data.get("timestamps")
        if ts_list is None:
            t = data.get("timestamp")
            if t is not None:
                ts_list = [t]
            else:
                ts_list = []
        all_ts.extend(ts_list)
    # Sort in place; timestamps should be datetime-like objects
    try:
        all_ts.sort()
    except TypeError:
        # If mixed types, convert to strings for stable ordering
        all_ts.sort(key=str)
    return all_ts


def _to_seconds(ts) -> float:
    """Convert a timestamp to a float (Unix epoch seconds) for arithmetic."""
    try:
        return ts.timestamp()
    except AttributeError:
        # pandas Timestamp
        return float(ts.value) / 1e9


def analyze_temporal_patterns(G_sub) -> Dict[str, bool]:
    """
    Detect temporal anomalies in a transaction subgraph.

    Parameters
    ----------
    G_sub : nx.DiGraph
        k-hop subgraph of the alert node (copy, not view).

    Returns
    -------
    {"burst": bool, "rapid_chain": bool}
    """
    all_ts = _all_timestamps(G_sub)

    burst = False
    rapid_chain = False

    if len(all_ts) < 2:
        return {"burst": burst, "rapid_chain": rapid_chain}

    # Convert to float seconds for arithmetic
    try:
        ts_sec = [_to_seconds(t) for t in all_ts]
    except Exception:
        # Fall back gracefully — cannot analyse
        return {"burst": burst, "rapid_chain": rapid_chain}

    # ------------------------------------------------------------------
    # Burst detection — sliding window
    # ------------------------------------------------------------------
    for i in range(len(ts_sec)):
        window_end = ts_sec[i] + _BURST_WINDOW_SEC
        count = sum(1 for t in ts_sec[i:] if t <= window_end)
        if count >= _BURST_MIN_TXS:
            burst = True
            break

    # ------------------------------------------------------------------
    # Rapid chain — sequence of length >= 3 where each gap <= 5 minutes
    # ------------------------------------------------------------------
    chain_len = 1
    for i in range(len(ts_sec) - 1):
        gap = ts_sec[i + 1] - ts_sec[i]
        if 0 <= gap <= _RAPID_GAP_SEC:
            chain_len += 1
            if chain_len >= _MIN_CHAIN_LEN:
                rapid_chain = True
                break
        else:
            chain_len = 1

    return {"burst": burst, "rapid_chain": rapid_chain}
