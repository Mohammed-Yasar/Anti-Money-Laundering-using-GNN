"""
main_investigation.py
---------------------
Investigation Agent pipeline for the AML detection system.

Pipeline:
    1. Load alerts from data/alerts/alerts.json
    2. Rebuild the TEST graph (same split used to produce the alerts)
    3. For each alert:
         expand_subgraph  →  detect_motifs  →  analyze_temporal_patterns
         →  generate_case_report
    4. Save all case reports to data/cases/cases.json
    5. Print investigation summary
"""

import json
import os
import time
import pandas as pd

from detection.split import load_and_split
from detection.graph_builder import build_graph

from investigation.subgraph import expand_subgraph
from investigation.motifs import detect_motifs
from investigation.temporal import analyze_temporal_patterns
from investigation.evidence import generate_case_report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_alerts(filepath: str) -> list:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_cases(cases: list, filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=4)


def print_section(title: str) -> None:
    print("\n" + "=" * 52)
    print(f"  {title}")
    print("=" * 52)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    start = time.time()

    # ------------------------------------------------------------------
    # 1. Load alerts
    # ------------------------------------------------------------------
    alerts_path = "data/alerts/alerts.json"
    if not os.path.exists(alerts_path):
        print(f"[ERROR] Alerts file not found: {alerts_path}")
        print("  Run `python main_gnn.py` first to generate alerts.")
        return

    alerts = load_alerts(alerts_path)
    print_section("INVESTIGATION AGENT — LOADING")
    print(f"  Alerts loaded: {len(alerts)}")

    # ------------------------------------------------------------------
    # 2. Rebuild the TEST graph (no leakage — same split as alert source)
    # ------------------------------------------------------------------
    print_section("REBUILDING TEST GRAPH")
    _, _, test_df = load_and_split()
    G_test, _ = build_graph(test_df)
    
    print(f"  Test graph: {G_test.number_of_nodes():,} nodes, "
          f"{G_test.number_of_edges():,} edges")

    # ------------------------------------------------------------------
    # 3. Investigation loop
    # ------------------------------------------------------------------
    print_section("RUNNING INVESTIGATION PER ALERT")
    cases = []
    missing_nodes = 0

    for idx, alert in enumerate(alerts, start=1):
        node_id = alert["node_id"]
        case_id = f"CASE_{idx:06d}"

        if idx <= 5:
            print(f"Node in graph: {node_id in G_test}")

        if node_id not in G_test:
            missing_nodes += 1
            continue
            
        # Expand k=2 hop subgraph (capped at 500 nodes internally)
        G_sub = expand_subgraph(G_test, node_id, k=2)

        # Detect motifs
        motifs = detect_motifs(G_sub)

        # Analyze temporal patterns
        temporal = analyze_temporal_patterns(G_sub)

        # Build case report
        report = generate_case_report(alert, G_sub, motifs, temporal, case_id)
        cases.append(report)

        if idx % 50 == 0 or idx == len(alerts):
            elapsed = time.time() - start
            print(f"  Processed {idx:>3}/{len(alerts)} alerts  "
                  f"({elapsed:.1f}s elapsed)")

    # ------------------------------------------------------------------
    # 4. Find similar cases
    # ------------------------------------------------------------------
    print_section("COMPUTING CASE SIMILARITIES")
    from investigation.evidence import find_similar_cases
    t0 = time.time()
    for case in cases:
        case["similar_cases"] = find_similar_cases(case, cases)
    print(f"  Similarity matching completed in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # 5. Save cases
    # ------------------------------------------------------------------
    cases_path = "data/cases/cases.json"
    save_cases(cases, cases_path)

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    smurfing_count = sum(1 for c in cases if c["motifs_detected"]["smurfing"])
    layering_count = sum(1 for c in cases if c["motifs_detected"]["layering"])
    circular_count = sum(1 for c in cases if c["motifs_detected"]["circular"])
    burst_count    = sum(1 for c in cases if c["temporal_anomalies"]["burst"])
    rapid_count    = sum(1 for c in cases if c["temporal_anomalies"]["rapid_chain"])

    print_section("INVESTIGATION SUMMARY")
    print(f"  TOTAL ALERTS          : {len(alerts)}")
    print(f"  TOTAL CASES GENERATED : {len(cases)}")
    if missing_nodes:
        print(f"  Skipped (node not in test graph): {missing_nodes}")
    print()
    print("  MOTIF DISTRIBUTION:")
    print(f"    SMURFING CASES  : {smurfing_count}")
    print(f"    LAYERING CASES  : {layering_count}")
    print(f"    CIRCULAR CASES  : {circular_count}")
    print()
    print("  TEMPORAL ANOMALIES:")
    print(f"    BURST           : {burst_count}")
    print(f"    RAPID CHAIN     : {rapid_count}")
    print()
    print(f"  Cases saved to: {cases_path}")
    print(f"  Total time: {time.time() - start:.1f}s")
    print("=" * 52)


if __name__ == "__main__":
    main()
