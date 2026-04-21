# AML-Sim: End-to-End Anti-Money Laundering Detection & Investigation Pipeline

A comprehensive, research-oriented pipeline for Anti-Money Laundering (AML) simulation, detection using Graph Neural Networks (GNNs), and automated investigations. This project provides a full-stack experience from synthetic data generation to interactive analyst dashboards.

## 🌟 Key Features

- **Realistic Synthetic Generation**: Simulates 90+ days of transactions with heavy-tailed account distributions (Barabási-Albert model) and realistic temporal noise.
- **Laundering Typologies**: Injects complex money laundering patterns including **Smurfing**, **Layering**, and **Circular (Round-tripping)**.
- **Graph-Based Detection**: Uses a 2-layer **Graph Attention Network (GAT)** for node-level risk prediction, outperforming traditional metric-based baselines.
- **Automated Investigation**: Features a sophisticated evidence engine that detects motifs, evaluates temporal signals, and builds consolidated "cases" for analysts.
- **Interactive Analyst Dashboard**: A Streamlit-based UI for reviewing alerts, exploring transaction subgraphs, and navigating case similarity.
- **Production-Ready Inference**: Supports saved model weights and feature scalers for rapid analysis of new datasets without retraining.

---

## 🏗 Project Architecture

```
aml_project/
├── data/
│   ├── raw/                # Generated/uploaded transaction datasets
│   ├── processed/          # Graphs and intermediate metrics
│   ├── alerts/             # Model-generated alerts (JSON)
│   ├── cases/              # Consolidated investigation cases (JSON)
│   └── models/             # Pretrained GNN weights and scalers
│
├── generator/              # Synthetic transaction generation engine
│   ├── config.py           # Generation parameters & seeds
│   ├── laundering.py       # Typology injection logic
│   └── temporal_engine.py  # Realistic transaction timing
│
├── gnn/                    # Graph Neural Network modules (PyTorch/PyG)
│   ├── model.py            # GAT Architecture
│   ├── trainer.py          # Training & probability extraction
│   └── data_builder.py     # Conversion to PyG Data objects
│
├── investigation/          # Alerting & Case discovery
│   ├── alert_queue.py      # Threshold-based alerting
│   ├── case_builder.py     # Evidence & motif extraction
│   └── motifs.py           # Structural pattern detection
│
├── detection/              # Core ML utilities
│   ├── graph_builder.py    # Transaction -> NetworkX conversion
│   ├── metrics.py          # Structural feature engineering
│   └── evaluation.py       # Precision/Recall & Typology analysis
│
├── pages/                  # Streamlit dashboard pages
├── app.py                  # Streamlit entry point
├── main_generate.py        # Generation Orchestrator
├── main_gnn.py             # GNN Pipeline Orchestrator
└── main_investigation.py   # Investigation Orchestrator
```

---

## 🚀 Getting Started

### 1. Installation

Requires Python 3.8+. It is recommended to use a virtual environment.

```bash
# Clone the repository
git clone https://github.com/your-repo/aml-sim.git
cd aml-sim

# Install dependencies
pip install -r requirements.txt
```

> [!NOTE]
> Ensure you have `torch` and `torch-geometric` installed. Refer to the [official PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) for CUDA-specific versions.

### 2. The AML Pipeline

The system is designed to run in a sequential pipeline:

**Step A: Generate Data**
Creates a synthetic transaction ecosystem with embedded laundering.
```bash
python main_generate.py
```

**Step B: Train & Detect**
Builds graphs, trains the GNN (or loads weights), and generates alerts in `data/alerts/`.
```bash
python main_gnn.py
```

**Step C: Investigate**
Analyzes alerts to build evidence-backed cases in `data/cases/`.
```bash
python main_investigation.py
```

### 3. Launch the Dashboard
View results and investigate cases in an interactive UI:
```bash
streamlit run app.py
```

---

## 🧠 Core Modules

### 🔍 GNN Detection Engine
The GNN treats accounts as nodes and transactions as edges. It computes structural features (PageRank, Betweenness, Triangles) and uses a **Graph Attention Network (GAT)** to learn illicit patterns.
- **Inference Mode**: Enable by setting `AML_INFERENCE=1` to run alerts on new data using stored weights without retraining.

### 🔬 Investigation Engine
Moves beyond simple scoring by performing:
- **Motif Detection**: Identifying smurfing stars or circular chains.
- **Temporal Analysis**: Detecting rapid "pass-through" behavior or burst activity.
- **Case Similarity**: Using case embeddings to find related investigation history.

---

## 📊 Dashboard Overview

The `app.py` dashboard provides four primary views:
1.  **Overview**: System-wide risk metrics and laundering distribution.
2.  **Alerts**: Prioritized list of high-risk nodes for review.
3.  **Investigation**: Interactive graph explorer for deep-diving into account neighborhoods.
4.  **Insights**: Typology-level performance and detection accuracy.

---

## 📜 License & Disclaimer
This project is for **educational and research purposes only**. It uses synthetic data generation and does not contain any real-world PII or financial data.

---
**Developed for Advanced AML Research.**
