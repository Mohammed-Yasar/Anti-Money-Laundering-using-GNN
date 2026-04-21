"""
detection/graph_builder.py
--------------------------
Build a directed NetworkX DiGraph from a transaction DataFrame.

Design principles:
    - One pass via pandas groupby for efficient edge aggregation.
    - Node attributes computed from aggregated sender / receiver stats.
    - All nodes appearing in the split are included (no zero-degree drops).
    - Edge attributes: count, total_amount, timestamps (sorted list).
    - Node attributes: in_tx_count, out_tx_count, total_tx_count,
                       in_amount, out_amount.
"""

from __future__ import annotations

import networkx as nx
import pandas as pd
from typing import Any


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_graph(
    df: pd.DataFrame,
) -> tuple[nx.DiGraph, dict[str, dict[str, Any]]]:
    """
    Build a directed transaction graph from a split DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Transaction DataFrame (one of train / val / test splits).
        Expected columns: sender, receiver, amount, timestamp.

    Returns
    -------
    G : nx.DiGraph
        Nodes = accounts; edges = aggregated transactions.
    node_stats : dict
        node_id -> {in_tx_count, out_tx_count, total_tx_count,
                    in_amount, out_amount}
    """
    if df.empty:
        return nx.DiGraph(), {}

    G = nx.DiGraph()

    # ------------------------------------------------------------------
    # 1. Collect all unique nodes
    # ------------------------------------------------------------------
    all_nodes = pd.unique(pd.concat([df["sender"], df["receiver"]]))
    G.add_nodes_from(all_nodes)

    # ------------------------------------------------------------------
    # 2. Aggregate edges via groupby (O(E log E), avoids Python loops)
    # ------------------------------------------------------------------
    edge_groups = (
        df
        .sort_values("timestamp")
        .groupby(["sender", "receiver"], sort=False)
    )

    edge_data: dict[tuple[str, str], dict[str, Any]] = {}
    for (sender, receiver), group in edge_groups:
        edge_data[(sender, receiver)] = {
            "count":        len(group),
            "total_amount": float(group["amount"].sum()),
            "timestamps":   group["timestamp"].tolist(),   # already sorted
        }

    G.add_edges_from(
        (src, dst, attrs)
        for (src, dst), attrs in edge_data.items()
    )

    # ------------------------------------------------------------------
    # 3. Compute per-node stats from sender / receiver aggregations
    # ------------------------------------------------------------------
    out_stats = (
        df.groupby("sender")
        .agg(out_tx_count=("amount", "count"), out_amount=("amount", "sum"))
        .rename_axis("node")
    )
    in_stats = (
        df.groupby("receiver")
        .agg(in_tx_count=("amount", "count"), in_amount=("amount", "sum"))
        .rename_axis("node")
    )

    node_df = (
        out_stats
        .join(in_stats, how="outer")
        .fillna(0)
        .astype({"out_tx_count": int, "in_tx_count": int})
    )
    node_df["total_tx_count"] = node_df["out_tx_count"] + node_df["in_tx_count"]

    # Attach as node attributes
    node_stats: dict[str, dict[str, Any]] = {}
    for node_id, row in node_df.iterrows():
        stats = {
            "in_tx_count":    int(row["in_tx_count"]),
            "out_tx_count":   int(row["out_tx_count"]),
            "total_tx_count": int(row["total_tx_count"]),
            "in_amount":      float(row["in_amount"]),
            "out_amount":     float(row["out_amount"]),
        }
        node_stats[node_id] = stats
        G.nodes[node_id].update(stats)

    # Nodes that appeared in the global node list but had zero transactions
    # (shouldn't happen for this dataset, but guard anyway)
    for node in G.nodes:
        if node not in node_stats:
            stats = {
                "in_tx_count": 0, "out_tx_count": 0,
                "total_tx_count": 0, "in_amount": 0.0, "out_amount": 0.0,
            }
            node_stats[node] = stats
            G.nodes[node].update(stats)

    return G, node_stats
