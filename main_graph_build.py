"""
main_graph_build.py
-------------------
Sprint 2A pipeline: load data, split, build graph, compute metrics, print summary.

Usage:
    python main_graph_build.py

This script does NOT:
    - Compute risk scores
    - Evaluate laundering_flag
    - Use any ML
    - Generate plots

It only verifies the graph + metric computation layer is working correctly.
"""

from __future__ import annotations

import time
from pprint import pformat

from detection.split import load_and_split
from detection.graph_builder import build_graph
from detection.metrics import compute_metrics


def print_section(title: str) -> None:
    """Print a formatted section header."""
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


def main() -> None:
    overall_start = time.time()

    # ------------------------------------------------------------------
    # STEP 1: Load and split
    # ------------------------------------------------------------------
    print_section("STEP 1: LOADING AND SPLITTING DATA")
    t0 = time.time()

    train_df, val_df, test_df = load_and_split()

    print(f"  Train split : {len(train_df):>8,} transactions  "
          f"(days 0-59)")
    print(f"  Val   split : {len(val_df):>8,} transactions  "
          f"(days 60-74)")
    print(f"  Test  split : {len(test_df):>8,} transactions  "
          f"(days 75-89)")
    print(f"  Split time  : {time.time() - t0:.2f}s")

    # Verify no leakage
    if not (
        train_df["day_index"].max() <= 59
        and val_df["day_index"].min() >= 60
        and val_df["day_index"].max() <= 74
        and test_df["day_index"].min() >= 75
    ):
        raise AssertionError("DATA LEAKAGE DETECTED – split boundaries violated!")
    print("  [PASS] No temporal leakage detected.")

    # ------------------------------------------------------------------
    # STEP 2: Build graph from training split
    # ------------------------------------------------------------------
    print_section("STEP 2: BUILDING TRAINING GRAPH")
    t0 = time.time()

    G, node_stats = build_graph(train_df)

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    avg_deg   = (2 * num_edges) / num_nodes if num_nodes > 0 else 0.0

    print(f"  Nodes       : {num_nodes:>8,}")
    print(f"  Edges       : {num_edges:>8,}")
    print(f"  Avg degree  : {avg_deg:>10.2f}")
    print(f"  Build time  : {time.time() - t0:.2f}s")

    # ------------------------------------------------------------------
    # STEP 3: Compute structural + temporal metrics
    # ------------------------------------------------------------------
    print_section("STEP 3: COMPUTING NODE METRICS")
    t0 = time.time()

    node_metrics = compute_metrics(G, node_stats)

    metric_time = time.time() - t0
    print(f"  Computed metrics for {len(node_metrics):,} nodes")
    print(f"  Metric time : {metric_time:.2f}s")

    import numpy as np
    
    # ------------------------------------------------------------------
    # STEP 4: Summary statistics across all nodes (TRAIN SPLIT)
    # ------------------------------------------------------------------
    print_section("STEP 4: METRIC DISTRIBUTION SUMMARY (TRAIN SPLIT)")

    if node_metrics:
        # Fanin
        fanins = [v["fanin"] for v in node_metrics.values()]
        print(f"  fanin               : mean={np.mean(fanins):8.3f}, std={np.std(fanins):8.3f}, max={np.max(fanins):8.3f}")

        # Fanout
        fanouts = [v["fanout"] for v in node_metrics.values()]
        print(f"  fanout              : mean={np.mean(fanouts):8.3f}, std={np.std(fanouts):8.3f}, max={np.max(fanouts):8.3f}")

        # Burst score
        bursts = [v["burst_score"] for v in node_metrics.values()]
        print(f"  burst_score         : mean={np.mean(bursts):8.3f}, p95={np.percentile(bursts, 95):8.3f}, max={np.max(bursts):8.3f}")

        # Chain score
        chains = [v["chain_score"] for v in node_metrics.values()]
        chain_pct_nz = (sum(1 for x in chains if x > 0) / len(chains)) * 100
        print(f"  chain_score         : % non-zero={chain_pct_nz:5.2f}%, max={np.max(chains):8.3f}")

        # Cycle score
        cycles = [v["cycle_score"] for v in node_metrics.values()]
        cycle_pct_nz = (sum(1 for x in cycles if x > 0) / len(cycles)) * 100
        print(f"  cycle_score         : % non-zero={cycle_pct_nz:5.2f}%, max={np.max(cycles):8.3f}")
    else:
        print("  No metrics computed.")

    # ------------------------------------------------------------------
    # STEP 5: Sample 5 node metrics
    # ------------------------------------------------------------------
    print_section("STEP 5: SAMPLE NODE METRICS (first 5 nodes)")

    sample_nodes = list(node_metrics.keys())[:5]
    for node_id in sample_nodes:
        stats   = node_stats.get(node_id, {})
        metrics = node_metrics[node_id]
        print(f"\n  Node: {node_id}")
        print(f"    in_tx_count    = {stats.get('in_tx_count', 0)}")
        print(f"    out_tx_count   = {stats.get('out_tx_count', 0)}")
        print(f"    total_tx_count = {stats.get('total_tx_count', 0)}")
        print(f"    in_amount      = {stats.get('in_amount', 0.0):.2f}")
        print(f"    out_amount     = {stats.get('out_amount', 0.0):.2f}")
        for k, v in metrics.items():
            print(f"    {k:<20} = {v:.4f}" if isinstance(v, float) else f"    {k:<20} = {v}")

    # ------------------------------------------------------------------
    # DONE
    # ------------------------------------------------------------------
    total_time = time.time() - overall_start
    print_section("PIPELINE COMPLETE")
    print(f"  Total runtime : {total_time:.2f}s")
    if total_time < 600:
        print("  [PASS] Completed within 10-minute runtime limit.")
    else:
        print("  [WARNING] Runtime exceeded 10-minute limit!")


if __name__ == "__main__":
    main()
