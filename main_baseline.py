"""
main_baseline.py
----------------
Sprint 2B end-to-end baseline pipeline:
    1. Load + split (no leakage)
    2. Build graph for train / val / test separately
    3. Compute structural metrics for each split
    4. Standardize using TRAIN stats only
    5. Score using fixed weights
    6. Normalize using TRAIN min/max (clip val/test to [min, max])
    7. Select threshold at 95th percentile of TRAIN scores
    8. Evaluate on val and test
    9. Print full structured report

Run:
    python main_baseline.py
"""

from __future__ import annotations

import time

from detection.split       import load_and_split
from detection.graph_builder import build_graph
from detection.metrics     import compute_metrics
from detection.baseline    import Standardizer, RiskScorer
from detection.evaluation  import (
    build_node_labels,
    evaluate,
    typology_recall,
    hub_fpr,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bar(title: str = "") -> None:
    print("=" * 48)
    if title:
        print(f"  {title}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    t_global = time.time()

    # ===================================================================
    # STEP 1: Load and split
    # ===================================================================
    _bar("STEP 1: LOADING AND SPLITTING DATA")
    train_df, val_df, test_df = load_and_split()
    print(f"  Train: {len(train_df):>8,} transactions")
    print(f"  Val  : {len(val_df):>8,} transactions")
    print(f"  Test : {len(test_df):>8,} transactions")

    # ===================================================================
    # STEP 2: Build graphs
    # ===================================================================
    _bar("STEP 2: BUILDING GRAPHS")
    t0 = time.time()

    G_train, ns_train = build_graph(train_df)
    G_val,   ns_val   = build_graph(val_df)
    G_test,  ns_test  = build_graph(test_df)

    print(f"  Train: {G_train.number_of_nodes():,} nodes, {G_train.number_of_edges():,} edges")
    print(f"  Val  : {G_val.number_of_nodes():,} nodes,  {G_val.number_of_edges():,} edges")
    print(f"  Test : {G_test.number_of_nodes():,} nodes,  {G_test.number_of_edges():,} edges")
    print(f"  Build time: {time.time() - t0:.1f}s")

    # ===================================================================
    # STEP 3: Compute metrics (per split, no shared state)
    # ===================================================================
    _bar("STEP 3: COMPUTING METRICS")
    t0 = time.time()

    m_train = compute_metrics(G_train, ns_train)
    m_val   = compute_metrics(G_val,   ns_val)
    m_test  = compute_metrics(G_test,  ns_test)

    print(f"  Train: {len(m_train):,} nodes")
    print(f"  Val  : {len(m_val):,} nodes")
    print(f"  Test : {len(m_test):,} nodes")
    print(f"  Metric time: {time.time() - t0:.1f}s")

    # ===================================================================
    # STEP 4: Standardize (fit on TRAIN only)
    # ===================================================================
    _bar("STEP 4: STANDARDIZING FEATURES")

    std = Standardizer()
    std.fit(m_train)                          # <-- TRAIN only

    z_train = std.transform(m_train)
    z_val   = std.transform(m_val)            # uses TRAIN mean/std
    z_test  = std.transform(m_test)           # uses TRAIN mean/std

    print("  [OK] Fitted on TRAIN. Applied to train / val / test.")

    # ===================================================================
    # STEP 5: Risk scores + normalisation + threshold (TRAIN only)
    # ===================================================================
    _bar("STEP 5: SCORING AND THRESHOLD SELECTION")

    scorer = RiskScorer()

    raw_train = RiskScorer.score(z_train)
    raw_val   = RiskScorer.score(z_val)
    raw_test  = RiskScorer.score(z_test)

    scorer.fit_normalizer(raw_train)          # <-- TRAIN only

    norm_train = scorer.normalize(raw_train)
    norm_val   = scorer.normalize(raw_val)    # clipped to [min_train, max_train]
    norm_test  = scorer.normalize(raw_test)   # clipped to [min_train, max_train]

    threshold = scorer.select_threshold(norm_train)   # <-- TRAIN only

    print(f"  Min (train): {scorer.min_train_:.4f}")
    print(f"  Max (train): {scorer.max_train_:.4f}")
    print(f"  Threshold (top 5% of TRAIN): {threshold:.4f}")

    # ===================================================================
    # STEP 6: Node labels (from each split's transactions only)
    # ===================================================================
    _bar("STEP 6: BUILDING NODE LABELS")

    labels_val  = build_node_labels(val_df)    # uses val_df only
    labels_test = build_node_labels(test_df)   # uses test_df only

    n_laun_val  = sum(labels_val.values())
    n_laun_test = sum(labels_test.values())
    print(f"  Val  laundering nodes: {n_laun_val:,} / {len(labels_val):,}")
    print(f"  Test laundering nodes: {n_laun_test:,} / {len(labels_test):,}")

    # ===================================================================
    # STEP 7: Flagged nodes
    # ===================================================================
    flagged_val  = {n for n, s in norm_val.items()  if s >= threshold}
    flagged_test = {n for n, s in norm_test.items() if s >= threshold}

    # ===================================================================
    # STEP 8: Evaluate
    # ===================================================================
    results_val  = evaluate(labels_val,  norm_val,  threshold)
    results_test = evaluate(labels_test, norm_test, threshold)

    # Typology recall — TEST only, split-local inference
    typ_recall = typology_recall(test_df, labels_test, flagged_test)

    # Hub FPR — TEST graph only
    hub_fpr_val  = hub_fpr(G_val,  flagged_val,  labels_val)
    hub_fpr_test = hub_fpr(G_test, flagged_test, labels_test)

    # ===================================================================
    # STEP 9: Print report
    # ===================================================================
    print()
    _bar()
    _bar("BASELINE RESULTS")
    _bar()

    print(f"\n  TRAIN threshold (top 5%) = {threshold:.4f}")

    print("\n  Validation:")
    print(f"    Precision = {results_val['precision']:.4f}")
    print(f"    Recall    = {results_val['recall']:.4f}")
    print(f"    F1        = {results_val['f1']:.4f}")
    print(f"    AUC       = {results_val['auc']:.4f}")
    print(f"    Hub FPR   = {hub_fpr_val:.4f}")

    print("\n  Test:")
    print(f"    Precision = {results_test['precision']:.4f}")
    print(f"    Recall    = {results_test['recall']:.4f}")
    print(f"    F1        = {results_test['f1']:.4f}")
    print(f"    AUC       = {results_test['auc']:.4f}")

    print("\n  Typology Recall (Test):")
    print(f"    Smurfing = {typ_recall.get('smurfing', 0.0):.4f}")
    print(f"    Layering = {typ_recall.get('layering', 0.0):.4f}")
    print(f"    Circular = {typ_recall.get('circular', 0.0):.4f}")

    print(f"\n  Hub False Positive Rate (Test)  = {hub_fpr_test:.4f}")

    _bar()
    print(f"\n  Total runtime: {time.time() - t_global:.1f}s")

    # Sanity check
    if results_val["f1"] > 0.75 or results_test["f1"] > 0.75:
        print("\n  [WARNING] F1 > 0.75 — potential label leakage. Investigate.")
    else:
        print("  [OK] Results within expected baseline range.")


if __name__ == "__main__":
    main()
