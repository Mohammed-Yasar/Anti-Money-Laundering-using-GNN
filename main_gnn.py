"""
main_gnn.py
-----------
Sprint 3A: Temporal GNN pipeline orchestrator.
1. Load data & splits
2. Compute structural metrics (features)
3. Prepare PyG Data objects
4. Train GAT model
5. Select 95th percentile TRAIN threshold
6. Evaluate & Compare
"""

import torch
import numpy as np
import time
import pandas as pd
import os
import json
import pickle

from detection.split import load_and_split
from detection.graph_builder import build_graph
from detection.metrics import compute_metrics
from detection.evaluation import build_node_labels, evaluate, typology_recall, hub_fpr
from detection.baseline import Standardizer, RiskScorer

from gnn.data_builder import build_pyg_data
from gnn.model import AMLGNN
from gnn.trainer import GNNTrainer

def print_section(title: str):
    print("\n" + "=" * 48)
    print(f"  {title}")
    print("=" * 48)

def main():
    start_global = time.time()
    
    is_inference = os.environ.get("AML_INFERENCE", "0") == "1"
    MODEL_PATH = "data/models/gnn_weights.pt"
    SCALER_PATH = "data/models/gnn_scaler.pkl"
    THRESH_PATH = "data/models/gnn_threshold.json"

    if is_inference:
        print_section("INFERENCE MODE ENABLED")
        print("  Generating alerts based on pretrained model...")

    # 1. Loading and Splitting
    print_section("STEP 1: LOADING AND SPLITTING DATA")
    train_df, val_df, test_df = load_and_split()
    print(f"  Train: {len(train_df):,} txs")
    print(f"  Val:   {len(val_df):,} txs")
    print(f"  Test:  {len(test_df):,} txs")

    # 2. Graph & Metrics (Features)
    print_section("STEP 2: BUILDING GRAPHS & COMPUTING METRICS")
    t0 = time.time()
    G_train, ns_train = build_graph(train_df)
    G_val,   ns_val   = build_graph(val_df)
    G_test,  ns_test  = build_graph(test_df)
    
    m_train = compute_metrics(G_train, ns_train)
    m_val   = compute_metrics(G_val,   ns_val)
    m_test  = compute_metrics(G_test,  ns_test)
    print(f"  Metrics computed in {time.time() - t0:.1f}s")

    # 3. PyG Data Building
    print_section("STEP 3: PREPARING GNN DATA")
    scaler = None
    if is_inference:
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
            
    data_train, data_val, data_test, node_to_idx, scaler = build_pyg_data(
        train_df, val_df, test_df,
        m_train, m_val, m_test,
        ns_train, ns_val, ns_test,
        scaler=scaler
    )
    print(f"  Nodes mapped: {len(node_to_idx):,}")
    print(f"  Test data: {data_test.x.shape[0]} nodes, {data_test.edge_index.shape[1]} edges")

    # 4. Training
    print_section(f"STEP 4: {'INFERENCE (PRETRAINED)' if is_inference else 'TRAINING GNN (2-LAYER GAT)'}")
    model = AMLGNN(in_dim=10)
    
    if is_inference:
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
        model.eval()
        print(f"  Loaded pretrained weights from {MODEL_PATH}")
        trainer = GNNTrainer(model)
        test_probs  = trainer.get_probs(data_test)
        
        active_scores = test_probs[data_test.mask.numpy()]
        threshold = float(np.percentile(active_scores, 97.5))
        print(f"  Dynamic Inference Threshold (97.5th percentile): {threshold:.4f}")
        print_section("GNN RESULTS (INFERENCE)")
        print("  No ground truth labels available \u2014 evaluation metrics disabled.")
    else:
        trainer = GNNTrainer(model, lr=0.005, patience=15, max_epochs=150)
        t0 = time.time()
        trainer.train(data_train, data_val)
        print(f"  Training completed in {time.time() - t0:.1f}s")
        
        # Save model state
        os.makedirs("data/models", exist_ok=True)
        torch.save(model.state_dict(), MODEL_PATH)
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(scaler, f)
            
        # 5. Threshold Selection (TRAIN Probabilities @ 95th Percentile)
        print_section("STEP 5: THRESHOLD SELECTION & EVALUATION")
        train_probs = trainer.get_probs(data_train)
        val_probs   = trainer.get_probs(data_val)
        test_probs  = trainer.get_probs(data_test)
        
        train_mask_np = data_train.mask.numpy()
        threshold = float(np.percentile(train_probs[train_mask_np], 95))
        print(f"  TRAIN Threshold (95th percentile): {threshold:.4f}")
        with open(THRESH_PATH, "w") as f:
            json.dump({"threshold": threshold}, f)

        # 6. Final Evaluation
        idx_to_node = {v: k for k, v in node_to_idx.items()}
        
        def get_results(data, probs, split_df, G_split):
            mask_np = data.mask.numpy()
            active_indices = np.where(mask_np)[0]
            
            # Prepare evaluation dicts
            # node_id -> prob/label for detection.evaluation functions
            split_labels = {idx_to_node[i]: float(data.y[i]) for i in active_indices}
            split_scores = {idx_to_node[i]: float(probs[i])  for i in active_indices}
            
            # Eval standard metrics
            metrics = evaluate(split_labels, split_scores, threshold)
            
            # Flagged nodes for typology/hub FPR
            flagged_nodes = {nid for nid, s in split_scores.items() if s >= threshold}
            
            return metrics, split_labels, flagged_nodes

        val_res, val_labels, val_flagged = get_results(data_val, val_probs, val_df, G_val)
        test_res, test_labels, test_flagged = get_results(data_test, test_probs, test_df, G_test)
        
        # Extended metrics on TEST
        typ_recall = typology_recall(test_df, test_labels, test_flagged)
        hub_fpr_test = hub_fpr(G_test, test_flagged, test_labels)

        # 7. Print Report
        print_section("GNN RESULTS")
        
        print("\n  Validation:")
        print(f"    Precision = {val_res['precision']:.4f}")
        print(f"    Recall    = {val_res['recall']:.4f}")
        print(f"    F1        = {val_res['f1']:.4f}")
        print(f"    AUC       = {val_res['auc']:.4f}")

        print("\n  Test:")
        print(f"    Precision = {test_res['precision']:.4f}")
        print(f"    Recall    = {test_res['recall']:.4f}")
        print(f"    F1        = {test_res['f1']:.4f}")
        print(f"    AUC       = {test_res['auc']:.4f}")

        print("\n  Typology Recall (Test):")
        for typ, rec in typ_recall.items():
            print(f"    {typ.capitalize():<10} = {rec:.4f}")

        print(f"\n  Hub False Positive Rate (Test) = {hub_fpr_test:.4f}")
        
        print("\n" + "=" * 48)
        print(f"  Pipeline completed in {time.time() - start_global:.1f}s")
        print("=" * 48)

        if test_res['auc'] >= 0.80 and test_res['f1'] >= 0.40:
            print("\n  [VERIFIED] GNN model reached expected performance gains.")
        elif test_res['auc'] > 0.90:
            print("\n  [WARNING] High AUC detected (>0.90). Check for leakage.")
    
    idx_to_node = {v: k for k, v in node_to_idx.items()}
    from investigation.alert_queue import build_alert_queue, save_alerts

    # Build node → score dictionary
    active_indices = np.where(data_test.mask.numpy())[0]
    test_scores = {idx_to_node[i]: float(test_probs[i]) for i in active_indices}

    alerts = build_alert_queue(test_scores, threshold)

    save_alerts(alerts, "data/alerts/alerts.json")

    print(f"\nGenerated {len(alerts)} investigation alerts.")
if __name__ == "__main__":
    main()
