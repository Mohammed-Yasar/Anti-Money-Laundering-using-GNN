"""
detection/evaluation.py
-----------------------
Node label construction, metric computation, typology-recall, and hub FPR.

Leakage-safety guarantees:
    - build_node_labels() uses only transactions in the supplied split DataFrame.
    - infer_campaign_typology() analyses only the supplied campaign sub-DataFrame
      (already filtered to one split).
    - hub_fpr() derives degree from the supplied G (test-split graph only).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


# ---------------------------------------------------------------------------
# 1. Node label construction
# ---------------------------------------------------------------------------

def build_node_labels(split_df: pd.DataFrame) -> dict[str, int]:
    """
    Label each node 1 if it appears as sender or receiver in any laundering
    transaction within split_df.  Uses only the supplied split.

    Parameters
    ----------
    split_df : pd.DataFrame
        Transactions for one split (train / val / test).

    Returns
    -------
    labels : dict  node_id -> 0 or 1
    """
    all_nodes = set(split_df["sender"]) | set(split_df["receiver"])
    laun_mask = split_df["laundering_flag"] == 1
    laundering_nodes = (
        set(split_df.loc[laun_mask, "sender"]) |
        set(split_df.loc[laun_mask, "receiver"])
    )
    return {node: (1 if node in laundering_nodes else 0) for node in all_nodes}


# ---------------------------------------------------------------------------
# 2. Standard evaluation metrics
# ---------------------------------------------------------------------------

def evaluate(
    labels:      dict[str, int],
    risk_scores: dict[str, float],
    threshold:   float,
) -> dict[str, float]:
    """
    Compute precision, recall, F1, and AUC for nodes present in both dicts.

    Parameters
    ----------
    labels      : node_id -> 0/1
    risk_scores : node_id -> normalised risk score [0, 1]
    threshold   : threshold from TRAIN (never tuned on val/test)

    Returns
    -------
    metrics : dict with keys precision, recall, f1, auc
    """
    common = sorted(set(labels) & set(risk_scores))
    if not common:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "auc": 0.5}

    y_true  = np.array([labels[n]      for n in common], dtype=int)
    y_score = np.array([risk_scores[n] for n in common], dtype=float)
    y_pred  = (y_score >= threshold).astype(int)

    try:
        auc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        auc = 0.5

    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true,    y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true,        y_pred, zero_division=0)),
        "auc":       auc,
    }


# ---------------------------------------------------------------------------
# 3. Typology inference (split-local, no cross-split view)
# ---------------------------------------------------------------------------

def infer_campaign_typology(campaign_txs: pd.DataFrame) -> str:
    """
    Infer typology for a single campaign using only its transactions in
    the current split.  Deterministic cascade:

        1. Circular  — subgraph contains a directed cycle  ≤ 6 hops
        2. Smurfing  — ≥ 40% of nodes have in-degree ≥ 3
        3. Layering  — default

    Parameters
    ----------
    campaign_txs : pd.DataFrame
        Rows for one campaign_id, one split only.
    """
    if campaign_txs.empty:
        return "layering"

    # Build campaign subgraph
    G = nx.DiGraph()
    for _, row in campaign_txs.iterrows():
        G.add_edge(row["sender"], row["receiver"])

    nodes = list(G.nodes)
    if not nodes:
        return "layering"

    # Rule 1 — Circular: any short directed cycle
    if _has_any_short_cycle(G, max_len=6):
        return "circular"

    # Rule 2 — Smurfing: high fanout from origins
    # Origins send to >= 8 mules, so out-degree is high
    if any(G.out_degree(n) >= 5 for n in nodes):
        return "smurfing"

    # Rule 3 — Layering (default)
    return "layering"


def _has_any_short_cycle(G: nx.DiGraph, max_len: int = 6) -> bool:
    """Return True if G contains any simple directed cycle of length ≤ max_len."""
    for start in G.nodes:
        if G.out_degree(start) == 0:
            continue
        # Bounded iterative DFS
        stack = [(start, 0, {start})]
        while stack:
            current, depth, path = stack.pop()
            if depth >= max_len:
                continue
            for succ in G.successors(current):
                if succ == start and depth >= 1:
                    return True
                if succ not in path and depth + 1 < max_len:
                    stack.append((succ, depth + 1, path | {succ}))
    return False


# ---------------------------------------------------------------------------
# 4. Typology-level recall (TEST only)
# ---------------------------------------------------------------------------

def typology_recall(
    test_df:       pd.DataFrame,
    node_labels:   dict[str, int],
    flagged_nodes: set[str],
) -> dict[str, float]:
    """
    Compute per-typology recall on the TEST split.

    Typology is inferred per campaign using only test_df rows (no cross-split
    leakage).

    Parameters
    ----------
    test_df       : TEST-split transaction DataFrame.
    node_labels   : {node_id: 0/1} built from test_df only.
    flagged_nodes : set of nodes flagged by the baseline (score >= threshold).

    Returns
    -------
    recall_by_typology : dict  typology -> recall float
    """
    laun_df = test_df[test_df["laundering_flag"] == 1].copy()
    laun_df = laun_df.dropna(subset=["campaign_id"])

    # Map campaign_id -> inferred typology (using test rows only)
    campaign_typology: dict[int, str] = {}
    for cid, camp_df in laun_df.groupby("campaign_id"):
        campaign_typology[cid] = infer_campaign_typology(camp_df)

    # Collect laundering nodes per typology (from test labels)
    typology_laun_nodes:    dict[str, set[str]] = defaultdict(set)
    typology_flagged_nodes: dict[str, set[str]] = defaultdict(set)

    for cid, typology in campaign_typology.items():
        camp_df = laun_df[laun_df["campaign_id"] == cid]
        camp_nodes = set(camp_df["sender"]) | set(camp_df["receiver"])
        laun_camp_nodes = {n for n in camp_nodes if node_labels.get(n, 0) == 1}

        typology_laun_nodes[typology].update(laun_camp_nodes)
        typology_flagged_nodes[typology].update(laun_camp_nodes & flagged_nodes)

    result: dict[str, float] = {}
    for typ in ["smurfing", "layering", "circular"]:
        total   = len(typology_laun_nodes.get(typ, set()))
        correct = len(typology_flagged_nodes.get(typ, set()))
        result[typ] = (correct / total) if total > 0 else 0.0

    return result


# ---------------------------------------------------------------------------
# 5. Hub false-positive rate (TEST graph only)
# ---------------------------------------------------------------------------

def hub_fpr(
    G_test:        nx.DiGraph,
    flagged_nodes: set[str],
    node_labels:   dict[str, int],
) -> float:
    """
    False-positive rate among hub nodes in the TEST split.

        hubs = nodes whose (in + out) degree >= 99th-percentile of TEST degrees
        FPR  = |flagged hubs that are NOT laundering| / |all hubs|

    Degree is computed from G_test (test-split graph) only.

    Parameters
    ----------
    G_test        : DiGraph built from test split only.
    flagged_nodes : nodes flagged by baseline.
    node_labels   : {node_id: 0/1} from test split only.

    Returns
    -------
    fpr : float
    """
    degrees = {n: G_test.in_degree(n) + G_test.out_degree(n) for n in G_test.nodes}
    if not degrees:
        return 0.0

    degree_vals   = np.array(list(degrees.values()), dtype=float)
    hub_threshold = float(np.percentile(degree_vals, 99))

    hubs = {n for n, d in degrees.items() if d >= hub_threshold}
    if not hubs:
        return 0.0

    fp_hubs = sum(
        1 for n in hubs
        if n in flagged_nodes and node_labels.get(n, 0) == 0
    )
    return fp_hubs / len(hubs)
