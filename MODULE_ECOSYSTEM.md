# AML Project Module Ecosystem

## Overview
A comprehensive Anti-Money Laundering system with synthetic data generation, graph neural network detection, and automated investigation capabilities. The pipeline flows through generation → detection → investigation.

---

## 1. GENERATOR Module
**Purpose**: Creates realistic synthetic transaction data with embedded money laundering patterns.

### 1.1 `config.py`
- **Purpose**: Central configuration repository for all generation parameters
- **Key Content**:
  - Account generation: `NUM_ACCOUNTS=40000`, hub/dormant ratios
  - Temporal parameters: `SIMULATION_DAYS=90`, transaction rates
  - Laundering: 350 campaigns with typology distributions (40% smurfing, 35% layering, 25% circular)
  - Network structure: Barabási-Albert model parameters
  - Validation thresholds and calibration parameters
- **Dependencies**: None (pure constants)
- **Usage**: Imported by all generator modules

### 1.2 `accounts.py`
- **Purpose**: Generates synthetic financial accounts with realistic activity profiles
- **Key Functions**:
  - `generate_accounts(seed)` → DataFrame with `[account_id, country_code, creation_day, activity_score]`
  - `_get_country_probabilities()` → Returns weighted distribution (US: 40%, IN: 25%, UK: 20%, SG: 10%, AE: 5%)
  - `save_accounts()` / `load_accounts()`
- **Key Classes**: None (functional module)
- **Dependencies**: numpy, pandas, config.py
- **Output**: Baseline account population with lognormal activity scores

### 1.3 `network.py`
- **Purpose**: Creates structural affinity graph representing account relationships
- **Key Functions**:
  - `create_structural_graph(account_ids, seed)` → NetworkX DiGraph using Barabási-Albert model
  - `compute_degree_statistics(graph)` → Returns mean/median/max degree and edge count
  - `get_neighbors(graph, node)` → Outgoing neighbors for transaction sender
  - `save_degree_statistics()`
- **Key Metrics**: Network follows power-law distribution (heavy tail for hubs)
- **Dependencies**: networkx, numpy, pandas, config.py
- **Output**: Directed graph for baseline transaction generation

### 1.4 `temporal_engine.py`
- **Purpose**: Generates realistic baseline (legitimate) transactions over 90-day simulation
- **Key Functions**:
  - `generate_baseline_transactions(accounts_df, graph, seed)` → List of baseline transaction dicts
  - `_get_volume_multiplier(day, date)` → Applies weekday/weekend/salary-day adjustments
  - `_is_salary_day(date)` → Heuristic for month-end activity spikes
  - `_generate_day_transactions()` → Processes single-day activity based on account activity scores
- **Temporal Logic**: 
  - Baseline transactions distributed across 90 days
  - ~4,500 transactions/day on average
  - Activity probabilities weighted by account activity_score
  - Potential receivers selected from structural graph neighbors
- **Dependencies**: numpy, pandas, networkx, datetime, config.py
- **Output**: List of transactions with `[sender, receiver, amount, timestamp, country_code]`

### 1.5 `laundering.py`
- **Purpose**: Injects three typologies of money laundering into account ecosystem
- **Key Functions**:
  - `inject_laundering_campaigns(accounts_df, graph, seed)` → Returns (laundering_txs, campaign_metadata)
  - `_select_laundering_nodes()` → Identifies 5% of nodes for illicit activity based on degree profile
  - `_generate_smurfing_campaign()` → 2-3 origins → 15-25 mules → 2-3 sinks pattern
  - `_generate_layering_campaign()` → 4-7 hop transfer chains
  - `_generate_circular_campaign()` → 5-7 hop cycles with branch transfers
  - `_get_degree_bins(graph)` → Classifies nodes as High/Mid/Low degree
- **Laundering Dynamics**:
  - 350 total campaigns across train/val/test splits
  - Node reuse capped at 20% to maintain campaign cohesion
  - Temporal compression: all typologies within 4-12 hour windows
  - Produces ~2-3% laundering ratio in final dataset
- **Dependencies**: numpy, pandas, networkx, datetime, config.py
- **Output**: Laundering transactions + metadata (campaign_id, typology, split assignment)

### 1.6 `transaction_builder.py`
- **Purpose**: Merges baseline and laundering transactions into final dataset
- **Key Functions**:
  - `build_final_dataset(baseline_txs, laundering_txs)` → Merged DataFrame
  - `_resolve_duplicate_timestamps()` → Ensures unique timestamps via microsecond offsets
  - `save_transactions()` / `compute_dataset_statistics()`
- **Output Format**: Columns include `[transaction_id, sender, receiver, amount, timestamp, sender_country, receiver_country, laundering_flag, campaign_id]`
- **Statistics**: Computes laundering ratio, temporal deltas, campaign sizes
- **Dependencies**: numpy, pandas, datetime
- **Output**: Final transaction CSV (~400k transactions)

---

## 2. DETECTION Module
**Purpose**: Converts transactions into graph representations and computes structural/temporal features for risk scoring.

### 2.1 `split.py`
- **Purpose**: Loads transaction CSV and creates train/val/test splits with no temporal leakage
- **Key Functions**:
  - `load_and_split(data_path)` → (train_df, val_df, test_df)
  - `_load(data_path, is_inference)` → Reads CSV with schema validation
  - `_add_day_index(df)` → Computes day offset from minimum timestamp
- **Split Strategy**:
  - Train: Days 0-59 (60 days)
  - Val: Days 60-74 (15 days)
  - Test: Days 75-89 (15 days)
- **Inference Mode**: When `AML_INFERENCE=1`, returns empty train/val and all data as test
- **Dependencies**: pandas, pathlib
- **Output**: Three DataFrames with temporal separation

### 2.2 `graph_builder.py`
- **Purpose**: Converts transaction DataFrame into NetworkX DiGraph with aggregated edge statistics
- **Key Functions**:
  - `build_graph(df)` → (nx.DiGraph, node_stats_dict)
- **Edge Aggregation**: Groups by (sender, receiver), stores:
  - `count`: transaction frequency
  - `total_amount`: aggregate volume
  - `timestamps`: sorted list of transaction times
- **Node Statistics** (per-node aggregation):
  - `in_tx_count`, `out_tx_count`, `total_tx_count`
  - `in_amount`, `out_amount` (in currency)
- **All Nodes Included**: No zero-degree nodes removed
- **Dependencies**: networkx, pandas
- **Output**: Transaction graph with rich edge/node metadata

### 2.3 `metrics.py`
- **Purpose**: Computes 6 structural/temporal metrics per node for ML feature engineering
- **Key Metrics**:
  - **fanin**: Unique incoming neighbors (predecessors)
  - **fanout**: Unique outgoing neighbors (successors)
  - **weighted_degree**: Total transaction count (in + out)
  - **burst_score**: Z-score of max transactions in any 1-hour window
  - **chain_score**: Approximate 3-hop participation (forward-looking temporal chains)
  - **cycle_score**: Binary indicator of participation in cycles ≤ 6 hops
- **Key Functions**:
  - `compute_metrics(G, node_stats)` → Dict[node_id → {6 metrics}]
  - `_build_timestamp_lookup()` → Pre-aggregates all timestamps per node
  - `_burst_score()` → Sliding window analysis
  - `_chain_score()` → Forward-window path detection
  - `_find_cycle_members()` → Pre-computes cycle membership
- **Performance Constraints**: O(N) node iteration, avoids O(N²) pair-wise comparisons
- **Dependencies**: networkx, numpy, itertools, collections
- **Output**: Node-level feature matrix

### 2.4 `baseline.py`
- **Purpose**: Standardizes features and computes risk scores (non-neural baseline)
- **Key Classes**:
  - **Standardizer**: Z-score normalization using only TRAIN statistics
    - `fit(train_metrics)` → Learns mean/std per feature
    - `transform(metrics)` → Applies z-score to any split
  - **RiskScorer**: Fixed-weight aggregation into [0,1] risk score
    - Fixed weights: fanin=0.30, weighted_degree=0.20, burst_score=0.20, cycle_score=0.15, fanout=0.10, chain_score=0.05
    - `score(z_metrics)` → Weighted sum
    - `fit_normalizer(train_scores)` → Learns min/max bounds
    - `normalize(scores)` → Clipped scaling to [0,1]
    - `select_threshold(train_norm_scores)` → 95th percentile
- **Leakage Prevention**: Normalization bounds learned only on TRAIN
- **Dependencies**: numpy
- **Output**: Normalized risk scores [0,1] + adaptive threshold

### 2.5 `evaluation.py`
- **Purpose**: Node labeling, metric computation, and typology-level analysis
- **Key Functions**:
  - `build_node_labels(split_df)` → Dict[node_id → {0, 1}] (1 if involved in any laundering tx)
  - `evaluate(labels, risk_scores, threshold)` → {precision, recall, f1, auc}
  - `infer_campaign_typology(campaign_txs)` → Deterministic cascade:
    1. Circular if any cycle ≤ 6 hops exists
    2. Smurfing if any node has out-degree ≥ 5
    3. Layering (default)
  - `typology_recall(test_df, node_labels, flagged_nodes)` → Per-typology recall
  - `hub_fpr(test_graph, node_labels, flagged_nodes)` → False positive rate on high-degree nodes
- **Leakage Safeguards**: All operations use only supplied split data
- **Dependencies**: networkx, numpy, pandas, sklearn.metrics
- **Output**: Standard metrics + typology-specific performance

---

## 3. GNN Module
**Purpose**: Neural network detection using Graph Attention Networks (PyTorch/PyG).

### 3.1 `model.py`
- **Purpose**: 2-layer Graph Attention Network architecture
- **Architecture**:
  - Layer 1: GAT with 4 heads, concat=True → output dim = 128
  - Layer 2: GAT with 1 head, concat=False → output dim = 32
  - Final linear projection → scalar logit per node
  - Activation: ELU between layers
  - Dropout: 0.3 per layer
- **Key Class**: `AMLGNN(nn.Module)`
  - `forward(x, edge_index)` → [Nodes] tensor of logits
- **Input Features**: 10 dimensional (from data_builder.py)
- **Output**: Node-level binary classification logits
- **Dependencies**: torch, torch.nn, torch_geometric.nn
- **Design**: Attention mechanism learns which edges are informative for risk prediction

### 3.2 `trainer.py`
- **Purpose**: Training loop with early stopping and checkpoint management
- **Key Class**: `GNNTrainer`
  - Constructor: Takes model, lr=0.005, weight_decay=1e-4, patience=15, max_epochs=150
  - `compute_pos_weight(y, mask)` → Imbalance correction for BCE loss
  - `train_epoch(data)` → Single epoch on TRAIN mask only
  - `evaluate(data)` → AUC score on specified mask (val or test)
  - `train(data_train, data_val)` → Full training with early stopping
  - `get_probs(data)` → Returns sigmoid probabilities for all nodes
- **Loss Function**: BCEWithLogitsLoss with position weighting for class imbalance
- **Early Stopping**: Tracks best val AUC, stops if no improvement for `patience` epochs
- **Optimization**: Adam optimizer
- **Dependencies**: torch, torch.optim, sklearn.metrics, numpy, copy
- **Output**: Trained model state + AUC metrics

### 3.3 `data_builder.py`
- **Purpose**: Converts split DataFrames + metrics into PyTorch Geometric Data objects
- **Key Function**:
  - `build_pyg_data(train_df, val_df, test_df, metrics dicts, node_stats dicts, scaler)` → (data_train, data_val, data_test, node_to_idx, scaler)
- **Feature Order** (10 features): fanin, fanout, weighted_degree, burst_score, chain_score, cycle_score, in_amount, out_amount, in_tx_count, out_tx_count
- **Processing Pipeline**:
  1. Global node indexing across all splits
  2. Feature matrix construction [N, 10]
  3. Edge aggregation (directed) [2, E]
  4. Label construction (1 if node in any laundering tx)
  5. Mask creation (1 for active nodes in split)
  6. Standardization (fit on TRAIN active nodes only, apply to all)
- **PyG Data Structure**: x (features), edge_index, y (labels), mask (active nodes)
- **Leakage Prevention**: Scaler fitted only on TRAIN, applied uniformly
- **Dependencies**: torch, numpy, pandas, torch_geometric.data, sklearn.preprocessing
- **Output**: Three PyG Data objects + global node mapping

---

## 4. INVESTIGATION Module
**Purpose**: Converts model predictions into actionable investigation cases with evidence.

### 4.1 `alert_queue.py`
- **Purpose**: Generates prioritized alert queue from GNN risk scores
- **Key Functions**:
  - `build_alert_queue(node_scores, threshold)` → List of alert dicts
    - Filters nodes with score ≥ threshold
    - Sorts descending by score
    - Assigns sequential alert_ids (AML_000001, ...)
    - Timestamp: UTC ISO format
  - `save_alerts(alerts, filepath)` → Writes JSON to data/alerts/alerts.json
- **Alert Structure**: {alert_id, node_id, risk_score, timestamp}
- **Dependencies**: os, json, datetime
- **Output**: Prioritized JSON alert queue

### 4.2 `subgraph.py`
- **Purpose**: Expands local neighborhood around alert nodes for investigation
- **Key Functions**:
  - `expand_subgraph(G, node_id, k=2)` → nx.DiGraph (k-hop induced subgraph copy)
  - `subgraph_stats(G_sub)` → {num_nodes, num_edges, density}
- **BFS Expansion**: 
  - Undirected traversal (captures both upstream and downstream)
  - Hard cap at MAX_SUBGRAPH_NODES=500 to avoid cycle-finding delays
  - Inference mode: More aggressive limits (40 nodes, 8 neighbors per node)
- **Return**: Induced subgraph copy (preserves full edge attributes)
- **Dependencies**: networkx, os
- **Output**: Local transaction neighborhood

### 4.3 `motifs.py`
- **Purpose**: Detects money laundering structural patterns within subgraphs
- **Key Functions**:
  - `detect_motifs(G_sub)` → {smurfing: bool, layering: bool, circular: bool}
  - `detect_smurfing(G)` → True if node has ≥4 incoming neighbors AND they connect to ≤3 unique sinks
  - `detect_layering(G)` → True if directed temporal chain of 3-5 hops with ≤60min gaps exists
  - `detect_circular(G)` → True if any simple cycle ≤6 hops exists
- **Temporal Constraints**: Layering requires monotonically increasing timestamps within 60min windows
- **Dependencies**: networkx
- **Output**: Binary motif indicators

### 4.4 `temporal.py`
- **Purpose**: Detects temporal anomalies in transaction patterns
- **Key Functions**:
  - `analyze_temporal_patterns(G_sub)` → {burst: bool, rapid_chain: bool}
  - `_burst_score()` → True if ≥5 transactions in any 1-hour window
  - `_rapid_chain()` → True if sequence of ≥3 txs with gaps ≤5 minutes
  - `_all_timestamps(G_sub)` → Aggregates and sorts all edge timestamps
- **Temporal Windows**: 
  - Burst: 3600 seconds (1 hour)
  - Rapid chain: 300 seconds (5 minutes)
- **Dependencies**: datetime operations
- **Output**: Temporal anomaly flags

### 4.5 `evidence.py`
- **Purpose**: Assembles investigation case reports with synthesized evidence
- **Key Functions**:
  - `generate_case_report(alert, subgraph, motif_results, temporal_results, case_id)` → Case dict
  - `build_case_vector(case)` → 8-dim float vector for similarity matching
  - `find_similar_cases(current_case, all_cases, top_k=3)` → List of similar case_ids via cosine similarity
- **Case Structure**:
  - Metadata: case_id, alert_id, node_id, risk_score
  - Typology: Inferred deterministically (circular > layering > smurfing > unknown)
  - Graph payload: Nodes + edges with timestamps
  - Explanation: Human-readable list built from detected signals only (no templates)
  - Motif/temporal results: Boolean flags
  - Similar cases: Top-k by cosine similarity
- **Leakage Prevention**: Each case uses only its own subgraph
- **Dependencies**: math, typing
- **Output**: Enriched JSON case dictionary

---

## 5. UTILITY & DASHBOARD

### 5.1 `utils/plotting.py`
- **Purpose**: Visualization helpers for validation and exploration
- **Key Functions**:
  - `plot_degree_distribution_loglog(graph, output_path)` → Saves log-log degree plot with power-law reference
  - `plot_daily_transaction_volume(df, output_path)` → Dual subplot: transaction count + volume over time
- **Output**: PNG files for validation reports
- **Dependencies**: matplotlib, seaborn, networkx, pandas, numpy

### 5.2 `app.py` (Streamlit Entry Point)
- **Purpose**: Multi-page interactive dashboard for AML investigation
- **Pages**:
  1. **1_Overview.py**: System-wide metrics, laundering distribution
  2. **2_Alerts.py**: Prioritized alert queue with risk scores
  3. **3_Investigation.py**: Interactive subgraph explorer with motif visualization
  4. **4_Insights.py**: Typology-level performance metrics
- **Capabilities**:
  - File uploader for custom transaction CSV
  - Pipeline orchestration (runs main_generate.py → main_gnn.py → main_investigation.py)
  - Inference mode toggle: `AML_INFERENCE=1` for new data
  - Session state management for node/case selection
- **Dependencies**: streamlit, pandas, subprocess

---

## 6. ORCHESTRATION SCRIPTS

### 6.1 Main Entry Points
- **main_generate.py**: Runs complete generation pipeline (generator modules)
- **main_gnn.py**: Loads data, trains GNN (or loads weights), generates alerts
- **main_investigation.py**: Builds case queue from alerts

---

## 7. DATA FLOW DIAGRAM

```
┌─────────────────────────────────────────────────────────────────┐
│                    GENERATOR PIPELINE                           │
├─────────────────────────────────────────────────────────────────┤
│  config.py                                                      │
│     ↓ (parameters)                                              │
│  accounts.py ─→ account_df [40k accounts]                      │
│  network.py  ─→ structural_graph [BA model]                    │
│     ↓                                                           │
│  temporal_engine.py ─→ baseline_txs [~395k legitimate]         │
│  laundering.py      ─→ laundering_txs [~5k illicit]            │
│     ↓                                                           │
│  transaction_builder.py ─→ transactions.csv [~400k rows]       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                        data/raw/transactions.csv
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    DETECTION PIPELINE                           │
├─────────────────────────────────────────────────────────────────┤
│  split.py ─→ train_df, val_df, test_df                         │
│     ↓ (no leakage)                                              │
│  graph_builder.py ─→ Graph_train, Graph_val, Graph_test        │
│     ↓                                                           │
│  metrics.py ─→ node_metrics (6 features per node)              │
│     ↓                                                           │
│  baseline.py ─→ risk_scores [non-neural baseline]              │
│     ↓                                                           │
│  OR: GNN detection                                              │
│  ├─ data_builder.py ─→ PyG Data objects                        │
│  ├─ model.py + trainer.py ─→ trained AMLGNN                    │
│  └─ GNN predictions ─→ risk_scores                             │
│     ↓                                                           │
│  evaluation.py ─→ labels, metrics, typology inference          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                       data/alerts/alerts.json
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                  INVESTIGATION PIPELINE                         │
├─────────────────────────────────────────────────────────────────┤
│  alert_queue.py ─→ sorted alerts (threshold-based)             │
│     ↓ (for each alert node)                                     │
│  subgraph.py ─→ k-hop neighborhood                             │
│     ↓                                                           │
│  ├─ motifs.py ─→ {smurfing, layering, circular}               │
│  ├─ temporal.py ─→ {burst, rapid_chain}                       │
│  └─ evidence.py ─→ case_report with explanation                │
│     ↓                                                           │
│  ├─ build_case_vector() ─→ 8-dim embedding                     │
│  └─ find_similar_cases() ─→ related investigations              │
│     ↓                                                           │
│  Save to data/cases/cases.json                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    app.py (Streamlit Dashboard)
                    ├─ 1_Overview.py
                    ├─ 2_Alerts.py
                    ├─ 3_Investigation.py
                    └─ 4_Insights.py
```

---

## 8. DEPENDENCY SUMMARY

### Core Libraries
- **numpy, pandas**: Data manipulation
- **networkx**: Graph operations
- **torch, torch_geometric**: GNN model & data structures
- **scikit-learn**: Metrics and preprocessing
- **matplotlib, seaborn**: Visualization
- **streamlit**: Web dashboard
- **datetime**: Temporal handling

### Module Interdependencies

| Module | Depends On | Used By |
|--------|-----------|---------|
| config.py | None | All generator modules |
| accounts.py | config, numpy, pandas | temporal_engine, laundering |
| network.py | config, networkx, numpy | temporal_engine, laundering |
| temporal_engine.py | accounts, network, config | transaction_builder |
| laundering.py | network, accounts, config | transaction_builder |
| transaction_builder.py | temporal_engine, laundering | main_generate.py |
| split.py | pandas | main_gnn.py |
| graph_builder.py | split, networkx, pandas | metrics.py |
| metrics.py | graph_builder, networkx, numpy | baseline.py, gnn/data_builder.py |
| baseline.py | metrics, numpy | evaluation.py |
| evaluation.py | baseline, split, networkx, pandas, sklearn | main_gnn.py |
| gnn/model.py | torch, torch_geometric | gnn/trainer.py |
| gnn/trainer.py | model, torch, sklearn | main_gnn.py |
| gnn/data_builder.py | metrics, split, evaluation, torch, sklearn | main_gnn.py |
| alert_queue.py | None | investigation/ |
| subgraph.py | networkx | motifs.py, temporal.py, evidence.py |
| motifs.py | networkx | evidence.py |
| temporal.py | datetime | evidence.py |
| evidence.py | motifs, temporal, typing | main_investigation.py |
| plotting.py | matplotlib, seaborn, networkx | validation |
| app.py | streamlit, pandas | Streamlit entry |

---

## 9. DESIGN PRINCIPLES

### Data Leakage Prevention
- Standardization bounds learned only on TRAIN
- Thresholds set only on TRAIN
- Metrics computed independently per split
- Typology inference uses only split's transactions

### Scalability
- Graph-level operations via NetworkX
- Pandas groupby for efficient aggregation
- Metric computation avoids O(N²) pairwise comparisons
- Subgraph capping (500 nodes) limits cycle detection overhead

### Modularity
- Each module has single responsibility
- Clear input/output contracts
- Config-driven parameterization
- Pluggable baseline vs GNN detection

### Interpretability
- Six interpretable metrics per node
- Deterministic typology inference (cascade)
- Evidence assembled from detected signals only
- Case similarity via cosine distance

---

## 10. KEY STATISTICS & CONFIGURATION

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Simulation Days | 90 | Full cycle for temporal splits |
| Accounts | 40,000 | Large network for heavy-tail effects |
| Base Transactions/Day | 4,500 | Realistic transaction volume |
| Laundering Campaigns | 350 | Embedded ground truth |
| Laundering Ratio | 2-3% | Realistic illicit ratio |
| Node Reuse Cap | 20% | Campaign cohesion |
| Train/Val/Test | 60/15/15 days | Temporal separation |
| GAT Layers | 2 (4-head + 1-head) | Attention-based learning |
| Feature Set | 10 features | Structural + temporal |
| Risk Score Threshold | 95th percentile | Top 5% flagged |
| Subgraph Hops | k=2 | Local + extended neighborhood |
| Max Subgraph Nodes | 500 | Computational efficiency |

