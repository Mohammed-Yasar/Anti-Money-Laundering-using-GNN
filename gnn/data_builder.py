"""
gnn/data_builder.py
-------------------
Builds PyG Data objects for TRAIN, VAL, and TEST splits.
Handles global node indexing, feature matrix construction, 
standardization (TRAIN-only fit), and labeling.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from detection.evaluation import build_node_labels

# Feature list in exact order
FEATURE_ORDER = [
    "fanin", "fanout", "weighted_degree", "burst_score", 
    "chain_score", "cycle_score", "in_amount", "out_amount", 
    "in_tx_count", "out_tx_count"
]

def build_pyg_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_metrics: Dict[str, Dict[str, Any]],
    val_metrics: Dict[str, Dict[str, Any]],
    test_metrics: Dict[str, Dict[str, Any]],
    train_node_stats: Dict[str, Dict[str, Any]],
    val_node_stats: Dict[str, Dict[str, Any]],
    test_node_stats: Dict[str, Dict[str, Any]],
    scaler=None,
) -> Tuple[Data, Data, Data, Dict[str, int], Any]:
    """
    Constructs PyG Data objects for all splits.
    
    Returns:
        data_train, data_val, data_test (PyG Data)
        node_to_idx (mapping)
        scaler (fitted StandardScaler)
    """
    
    # 1. Global Node Index
    all_senders = pd.concat([train_df["sender"], val_df["sender"], test_df["sender"]])
    all_receivers = pd.concat([train_df["receiver"], val_df["receiver"], test_df["receiver"]])
    all_nodes = sorted(pd.unique(pd.concat([all_senders, all_receivers])))
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}
    num_nodes = len(all_nodes)

    def get_split_data(df, metrics, node_stats):
        # A. Feature Matrix X [N, 10]
        X = np.zeros((num_nodes, len(FEATURE_ORDER)), dtype=np.float32)
        
        # Merge metrics and node_stats for convenience
        combined_features = {}
        # Metrics contains: fanin, fanout, weighted_degree, burst_score, chain_score, cycle_score
        # node_stats contains: in_amount, out_amount, in_tx_count, out_tx_count, total_tx_count
        for node_id in metrics:
            combined_features[node_id] = {**metrics[node_id], **node_stats[node_id]}
            
        for node_id, feats in combined_features.items():
            if node_id in node_to_idx:
                idx = node_to_idx[node_id]
                for f_idx, feat_name in enumerate(FEATURE_ORDER):
                    X[idx, f_idx] = feats.get(feat_name, 0.0)
                    
        # B. Edge Index [2, E]
        # Aggregate edges (similar to graph_builder but mapped to global idx)
        edges = df.groupby(["sender", "receiver"]).size().reset_index()
        edge_index = torch.tensor([
            [node_to_idx[s] for s in edges["sender"]],
            [node_to_idx[r] for r in edges["receiver"]]
        ], dtype=torch.long)
        
        # C. Labels y [N]
        labels_dict = build_node_labels(df)
        y = torch.zeros(num_nodes, dtype=torch.float)
        for node_id, label in labels_dict.items():
            y[node_to_idx[node_id]] = float(label)
            
        # D. Mask (Boolean)
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        for node_id in metrics: # active in this split
            mask[node_to_idx[node_id]] = True
            
        return X, edge_index, y, mask

    # Get raw features for all splits
    X_train_raw, edge_train, y_train, mask_train = get_split_data(train_df, train_metrics, train_node_stats)
    X_val_raw,   edge_val,   y_val,   mask_val   = get_split_data(val_df,   val_metrics,   val_node_stats)
    X_test_raw,  edge_test,  y_test,  mask_test  = get_split_data(test_df,  test_metrics,  test_node_stats)

    # 2. Standardization
    if scaler is None:
        scaler = StandardScaler()
        # Fit only on active nodes in TRAIN split
        active_train_idx = np.where(mask_train.numpy())[0]
        if len(active_train_idx) > 0:
            scaler.fit(X_train_raw[active_train_idx])
        else:
            # Fallback if train is empty (should only happen in inference if something weird happens)
            scaler.fit(X_train_raw)
            
    X_train = torch.from_numpy(scaler.transform(X_train_raw))
    X_val   = torch.from_numpy(scaler.transform(X_val_raw))
    X_test  = torch.from_numpy(scaler.transform(X_test_raw))

    data_train = Data(x=X_train, edge_index=edge_train, y=y_train, mask=mask_train)
    data_val   = Data(x=X_val,   edge_index=edge_val,   y=y_val,   mask=mask_val)
    data_test  = Data(x=X_test,  edge_index=edge_test,  y=y_test,  mask=mask_test)

    return data_train, data_val, data_test, node_to_idx, scaler
