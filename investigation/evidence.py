"""
investigation/evidence.py
--------------------------
Case report generation for the AML investigation layer.

Each report is assembled deterministically from detected signals only.
No static templates or random explanations are used.
"""

import math
from typing import Any, Dict, List


def generate_case_report(
    alert: Dict[str, Any],
    subgraph,
    motif_results: Dict[str, bool],
    temporal_results: Dict[str, bool],
    case_id: str,
) -> Dict[str, Any]:
    """
    Assemble a structured investigation case report.

    Parameters
    ----------
    alert            : alert dict from alerts.json
    subgraph         : nx.DiGraph — k-hop subgraph copy
    motif_results    : output of detect_motifs()
    temporal_results : output of analyze_temporal_patterns()
    case_id          : sequential identifier, e.g. "CASE_000001"

    Returns
    -------
    dict — structured case report
    """
    # ------------------------------------------------------------------
    # Human-readable explanation — built from detected signals only
    # ------------------------------------------------------------------
    explanation: List[str] = []

    if motif_results.get("smurfing"):
        explanation.append("high fan-in from multiple sources to a common sink")

    if motif_results.get("layering"):
        explanation.append("multi-hop transfer chain detected")

    if motif_results.get("circular"):
        explanation.append("circular money flow detected")

    if temporal_results.get("burst"):
        explanation.append("suspicious burst activity within 1-hour window")

    if temporal_results.get("rapid_chain"):
        explanation.append("rapid transaction chain with short inter-hop delays")

    if not explanation:
        explanation.append("elevated risk score exceeds detection threshold")

    # ------------------------------------------------------------------
    # Explicit typology precedence
    # ------------------------------------------------------------------
    if motif_results.get("circular"):
        typology = "circular"
    elif motif_results.get("layering"):
        typology = "layering"
    elif motif_results.get("smurfing"):
        typology = "smurfing"
    else:
        typology = "unknown"

    # ------------------------------------------------------------------
    # Serialize the graph payload
    # ------------------------------------------------------------------
    nodes_payload = list(subgraph.nodes())
    edges_payload = []
    
    for u, v, data in subgraph.edges(data=True):
        raw_ts = data.get("timestamps", [])
        clean_ts = [str(ts) for ts in raw_ts]
        
        edges_payload.append({
            "source": u,
            "target": v,
            "amount": data.get("amount", 0.0),
            "timestamps": clean_ts
        })
        
    # ------------------------------------------------------------------
    # Assemble report
    # ------------------------------------------------------------------
    return {
        "case_id":            case_id,
        "alert_id":           alert["alert_id"],
        "node_id":            alert["node_id"],
        "risk_score":         alert["risk_score"],
        "typology":           typology,
        "subgraph_nodes":     subgraph.number_of_nodes(),
        "subgraph_edges":     subgraph.number_of_edges(),
        "motifs_detected":    motif_results,
        "temporal_anomalies": temporal_results,
        "similar_cases":      [],
        "explanation":        explanation,
        "graph": {
            "nodes": nodes_payload,
            "edges": edges_payload
        }
    }


def build_case_vector(case: Dict[str, Any]) -> List[float]:
    """Build a numeric vector for similarity matching."""
    return [
        float(case["motifs_detected"]["smurfing"]),
        float(case["motifs_detected"]["layering"]),
        float(case["motifs_detected"]["circular"]),
        float(case["temporal_anomalies"]["burst"]),
        float(case["temporal_anomalies"]["rapid_chain"]),
        float(case["subgraph_nodes"]) / 500.0,
        float(case["subgraph_edges"]) / 500.0,
        float(case["risk_score"])
    ]


def find_similar_cases(current_case: Dict[str, Any], all_cases: List[Dict[str, Any]], top_k: int = 3) -> List[str]:
    """Find the top-k most similar cases using safe cosine similarity."""
    vec_a = build_case_vector(current_case)
    norm_a = math.sqrt(sum(x * x for x in vec_a))
    norm_a = max(norm_a, 1e-8)
    
    similarities = []
    
    for case in all_cases:
        if case["case_id"] == current_case["case_id"]:
            continue
            
        vec_b = build_case_vector(case)
        norm_b = math.sqrt(sum(x * x for x in vec_b))
        norm_b = max(norm_b, 1e-8)
        
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        sim = dot / (norm_a * norm_b)
        
        similarities.append((sim, case["case_id"]))
        
    # Sort descending by similarity
    similarities.sort(key=lambda x: x[0], reverse=True)
    
    return [c_id for sim, c_id in similarities[:top_k]]
