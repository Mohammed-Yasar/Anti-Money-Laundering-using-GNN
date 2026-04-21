"""
Alert Queue Module for AML Investigation System

Handles the conversion of GNN risk scores into actionable investigation alerts.
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Any


def build_alert_queue(node_scores: Dict[str, float], threshold: float) -> List[Dict[str, Any]]:
    """
    Converts node risk scores into a sorted list of investigation alerts.
    
    Args:
        node_scores (dict): Dictionary mapping node_id -> risk_score.
        threshold (float): Minimum risk score to trigger an alert.
        
    Returns:
        list: Sorted list of alert dictionaries (descending by risk_score).
    """
    alerts = []
    # Generate timestamp using datetime.utcnow()
    timestamp = datetime.utcnow().isoformat() + "Z"
    
    # Filter and sort the nodes based on the risk score threshold (descending order)
    filtered_nodes = [(k, v) for k, v in node_scores.items() if v >= threshold]
    filtered_nodes.sort(key=lambda x: x[1], reverse=True)
    
    # Create the alert entries with sequentially incrementing alert_ids
    for idx, (node_id, risk_score) in enumerate(filtered_nodes, start=1):
        alert_id = f"AML_{idx:06d}"
        alerts.append({
            "alert_id": alert_id,
            "node_id": node_id,
            "risk_score": risk_score,
            "timestamp": timestamp
        })
        
    return alerts


def save_alerts(alerts: List[Dict[str, Any]], filepath: str) -> None:
    """
    Saves a list of alerts to a JSON file.
    
    Args:
        alerts (list): The list of alert dictionaries to save.
        filepath (str): The destination file path.
    """
    # Ensure the directory exists
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)
        
    # Pretty format JSON and save
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(alerts, f, indent=4)


if __name__ == "__main__":
    # Dummy scores for testing
    dummy_scores = {
        "ACC0001": 0.21,
        "ACC0002": 0.87,
        "ACC0003": 0.64,
        "ACC0004": 0.98,
        "ACC0005": 0.81
    }
    
    # 95th percentile-like threshold definition for dummy data
    test_threshold = 0.80
    
    print(f"Building alerts with threshold >= {test_threshold}...\n")
    queued_alerts = build_alert_queue(dummy_scores, test_threshold)
    
    for alt in queued_alerts:
        print(f"Alert: {alt['alert_id']} | Node: {alt['node_id']} | Risk: {alt['risk_score']} | Time: {alt['timestamp']}")
        
    # Save the alerts to JSON
    output_tgt = "data/alerts/alerts.json"
    print(f"\nSaving generated alerts to {output_tgt}...")
    save_alerts(queued_alerts, output_tgt)
    print("Execution complete.")
