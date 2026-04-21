import os
import json
import random
import pandas as pd
import numpy as np
import networkx as nx

from detection.split import load_and_split
from detection.graph_builder import build_graph
from investigation.subgraph import expand_subgraph
from detection.metrics import compute_metrics

def main():
    # Detect if we are in inference mode (uploaded dataset)
    is_inference = os.path.exists("data/raw/uploaded_transactions.csv")
    if is_inference:
        os.environ["AML_INFERENCE"] = "1"
        os.environ["AML_DATA_PATH"] = "data/raw/uploaded_transactions.csv"
        
    print("=== DATASET ===")
    train_df, val_df, test_df = load_and_split()
    
    # The dataset being used depends on mode. For investigation, it's test_df.
    df = test_df
    print(f"Total transactions: {len(df)}")
    senders = set(df["sender"].unique())
    receivers = set(df["receiver"].unique())
    print(f"Unique sender count: {len(senders)}")
    print(f"Unique receiver count: {len(receivers)}")
    
    nodes = senders.union(receivers)
    num_nodes = len(nodes)
    print(f"Total unique nodes: {num_nodes}")
    if num_nodes > 0:
        avg_tx = len(df) / num_nodes
        deg_proxy = (2 * len(df)) / num_nodes
    else:
        avg_tx = 0
        deg_proxy = 0
    print(f"Average transactions per node: {avg_tx:.2f}")
    print(f"Degree proxy = (2 * num_tx) / num_nodes: {deg_proxy:.2f}")
    
    all_entities = pd.concat([df["sender"], df["receiver"]])
    top5_tx = all_entities.value_counts().head(5)
    print("\nTop 5 nodes by transaction count:")
    for n, count in top5_tx.items():
        print(f"  {n}: {count}")

    print("\n=== GRAPH ===")
    G_eval, node_stats = build_graph(df)
    
    print(f"Number of nodes in graph: {G_eval.number_of_nodes()}")
    print(f"Number of edges: {G_eval.number_of_edges()}")
    
    if G_eval.number_of_nodes() > 0:
        degrees = [d for n, d in G_eval.degree()]
        avg_deg = np.mean(degrees)
        max_deg = np.max(degrees)
        print(f"Average degree: {avg_deg:.2f}")
        print(f"Max degree: {max_deg}")
        if avg_deg > 0:
            print(f"Max / mean degree ratio: {max_deg / avg_deg:.2f}")
        else:
            print(f"Max / mean degree ratio: 0")
            
        print("\nRandom nodes:")
        random_nodes = random.sample(list(G_eval.nodes()), min(3, len(G_eval.nodes())))
        for n in random_nodes:
            neighbors = list(G_eval.neighbors(n)) + list(G_eval.predecessors(n))
            neighbors = list(set(neighbors))
            print(f"  Node ID: {n}")
            print(f"  Number of neighbors: {len(neighbors)}")
            print(f"  First 5 neighbors: {neighbors[:5]}")
    else:
        print("Graph is empty.")

    print("\n=== ALERT VALIDATION ===")
    try:
        with open("data/alerts/alerts.json", "r") as f:
            alerts = json.load(f)
    except FileNotFoundError:
        alerts = []
        print("No alerts.json found.")
        
    first_10 = alerts[:10]
    for a in first_10:
        nid = a["node_id"]
        in_graph = nid in G_eval
        deg = G_eval.degree(nid) if in_graph else "N/A"
        
        # calculate tx involving node
        tx_count = sum(all_entities == nid) if in_graph else 0
        
        print(f"Alert Node: {nid}")
        print(f"  Check if node_id exists in graph (True/False): {in_graph}")
        print(f"  Degree of node (if exists): {deg}")
        print(f"  Number of transactions involving node: {tx_count}")

    print("\n=== SUBGRAPH CHECK ===")
    counts = []
    for a in first_10:
        nid = a["node_id"]
        if nid in G_eval:
            G_sub = expand_subgraph(G_eval, nid, k=2)
            n_count = G_sub.number_of_nodes()
            e_count = G_sub.number_of_edges()
            counts.append((n_count, e_count))
            first_5 = list(G_sub.nodes())[:5]
            print(f"Alert Node: {nid}")
            print(f"  subgraph_nodes count: {n_count}")
            print(f"  subgraph_edges count: {e_count}")
            print(f"  First 5 node IDs in subgraph: {first_5}")
        else:
            print(f"Alert Node: {nid} (Not in graph)")
    
    if counts:
        all_same = len(set(counts)) == 1
        print(f"\nAll subgraphs same size: {all_same}")
    else:
        print("\nAll subgraphs same size: N/A")

    print("\n=== EDGE ATTRIBUTES ===")
    if G_eval.number_of_edges() > 0:
        edges = list(G_eval.edges(data=True))
        random_edges = random.sample(edges, min(5, len(edges)))
        for u, v, d in random_edges:
            print(f"edge ({u} -> {v})")
            print(f"attributes (full dict): {d}")
            has_timestamps = "timestamps" in d
            has_timestamp = "timestamp" in d
            print(f"  Has 'timestamps': {has_timestamps}")
            print(f"  Has 'timestamp': {has_timestamp}")
    else:
        print("No edges in graph.")

    print("\n=== FEATURES ===")
    if G_eval.number_of_nodes() > 0:
        metrics = compute_metrics(G_eval, node_stats)
        burst_scores = [m["burst_score"] for m in metrics.values()]
        chain_scores = [m["chain_score"] for m in metrics.values()]
        cycle_scores = [m["cycle_score"] for m in metrics.values()]
        
        print(f"burst_score mean: {np.mean(burst_scores):.4f}")
        print(f"burst_score std: {np.std(burst_scores):.4f}")
        print(f"burst_score max: {np.max(burst_scores):.4f}")
        
        pct_chain = sum(x > 0 for x in chain_scores) / len(chain_scores) * 100
        print(f"chain_score % non-zero: {pct_chain:.2f}%")
        
        pct_cycle = sum(x > 0 for x in cycle_scores) / len(cycle_scores) * 100
        print(f"cycle_score % non-zero: {pct_cycle:.2f}%")
        
        # top 5 nodes by burst_score
        sorted_burst = sorted(metrics.items(), key=lambda x: x[1]["burst_score"], reverse=True)
        print("\nTop 5 nodes by burst_score:")
        for nid, vals in sorted_burst[:5]:
            print(f"  {nid}: {vals['burst_score']:.4f}")
    else:
        print("No nodes to compute features.")

    print("\n=== ALERT SCORES ===")
    if alerts:
        scores = [a["risk_score"] for a in alerts]
        print(f"min risk_score: {np.min(scores):.4f}")
        print(f"max risk_score: {np.max(scores):.4f}")
        print(f"mean risk_score: {np.mean(scores):.4f}")
        
        sorted_alerts = sorted(alerts, key=lambda a: a["risk_score"], reverse=True)
        print("\nTop 5 highest risk nodes:")
        for a in sorted_alerts[:5]:
            print(f"  {a['node_id']}: {a['risk_score']:.4f}")
    else:
        print("No alerts to analyze scores.")
        
    print("\n=== GRAPH CONSISTENCY ===")
    if is_inference:
        print("GNN graph built from: Data passed via inference mode (which defaults to full df as test_df)")
    else:
        print("GNN graph built from: train_df, val_df, test_df splits separately.")
    print("Investigation graph built from: test_df")


if __name__ == "__main__":
    main()
