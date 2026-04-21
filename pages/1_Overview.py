"""
pages/1_Overview.py — System Summary Page
"""
import json
import sys
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.title("📊 System Overview")
st.markdown("High-level summary of the AML detection system.")

# Check Inference Mode status
is_inf_mode = os.path.exists("data/models/gnn_weights.pt")
if is_inf_mode:
    st.info("💡 **Inference Mode Active**: Dashboard is using a pretrained GNN model.")
else:
    st.warning("⚠️ **Training Mode**: No pretrained model found. System will train on next run.")

# ── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_transactions():
    path_up = "data/raw/uploaded_transactions.csv"
    path_tx = "data/raw/transactions.csv"
    path = path_up if os.path.exists(path_up) else path_tx
    
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str)

@st.cache_data
def load_alerts():
    with open("data/alerts/alerts.json", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_cases():
    with open("data/cases/cases.json", encoding="utf-8") as f:
        return json.load(f)

tx_df     = load_transactions()
alerts    = load_alerts()
cases     = load_cases()
cases_df  = pd.DataFrame(cases)

# ── KPI Computation ───────────────────────────────────────────────────────────
total_tx      = len(tx_df) if not tx_df.empty else "N/A"
total_accounts = len(
    set(tx_df["sender"].dropna().tolist() + tx_df["receiver"].dropna().tolist())
) if not tx_df.empty else "N/A"
total_alerts  = len(alerts)
total_cases   = len(cases)

# Laundering ratio from transactions CSV
if not tx_df.empty and "laundering_flag" in tx_df.columns:
    flagged       = (tx_df["laundering_flag"].astype(int) == 1).sum()
    launder_ratio = round(flagged / len(tx_df) * 100, 2)
else:
    launder_ratio = "N/A"

# ── KPI Cards ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)

def kpi_card(col, label, value, color="#1f77b4"):
    col.markdown(
        f"""
        <div style="
            background-color:#1a1a2e;
            border-left:4px solid {color};
            border-radius:6px;
            padding:16px 12px;
            text-align:center;
        ">
            <p style="color:#aaa;font-size:12px;margin:0;">{label}</p>
            <p style="color:white;font-size:26px;font-weight:700;margin:4px 0 0 0;">{value:,}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

kpi_card(c1, "Total Transactions", total_tx, "#0099ff")
kpi_card(c2, "Total Accounts",     total_accounts, "#00cc88")
kpi_card(c3, "Total Alerts",       total_alerts, "#ff6b35")
kpi_card(c4, "Total Cases",        total_cases, "#9b59b6")
c5.markdown(
    f"""
    <div style="
        background-color:#1a1a2e;
        border-left:4px solid #e74c3c;
        border-radius:6px;
        padding:16px 12px;
        text-align:center;
    ">
        <p style="color:#aaa;font-size:12px;margin:0;">Laundering Ratio</p>
        <p style="color:#e74c3c;font-size:26px;font-weight:700;margin:4px 0 0 0;">{launder_ratio}%</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)

if total_alerts == 0 and not tx_df.empty:
    st.success("✅ **No alerts found**: The current model did not detect any high-risk activity in this dataset.")
elif total_alerts > 0:
    st.warning(f"🚨 **Attention**: {total_alerts} suspicious nodes have been flagged for investigation.")

# ── Charts ────────────────────────────────────────────────────────────────────
left, right = st.columns(2)

with left:
    st.subheader("Typology Distribution")
    if "typology" in cases_df.columns:
        typo_counts = cases_df["typology"].value_counts().reset_index()
        typo_counts.columns = ["Typology", "Count"]
        fig_typo = px.bar(
            typo_counts, x="Typology", y="Count",
            color="Typology",
            color_discrete_sequence=px.colors.qualitative.Bold,
            template="plotly_dark",
        )
        fig_typo.update_layout(showlegend=False, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_typo, use_container_width=True)
    else:
        st.info("Typology data not available.")

with right:
    st.subheader("Temporal Anomaly Counts")
    if "temporal_anomalies" in cases_df.columns:
        temporal_df = pd.json_normalize(cases_df["temporal_anomalies"])
        anomaly_counts = temporal_df.sum().reset_index()
        anomaly_counts.columns = ["Anomaly", "Count"]
        fig_anom = px.bar(
            anomaly_counts, x="Anomaly", y="Count",
            color="Anomaly",
            color_discrete_sequence=["#ff6b35", "#9b59b6"],
            template="plotly_dark",
        )
        fig_anom.update_layout(showlegend=False, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_anom, use_container_width=True)
    else:
        st.info("Temporal anomaly data not available.")

st.markdown("<br>", unsafe_allow_html=True)

# ── Dataset Quality Validation Panel ─────────────────────────────────────────
st.subheader("🗂 Dataset Quality Check")

_uploaded_path = "data/raw/uploaded_transactions.csv"
_default_path  = "data/raw/transactions.csv"

if os.path.exists(_uploaded_path):
    _qdf = pd.read_csv(_uploaded_path)
    st.caption("Source: uploaded file")
elif os.path.exists(_default_path):
    _qdf = pd.read_csv(_default_path, dtype=str)
    st.caption("Source: default transactions dataset")
else:
    st.info("No transaction data found — upload a CSV or run the generator.")
    _qdf = None

if _qdf is not None:
    num_transactions = len(_qdf)
    num_nodes = len(set(_qdf["sender"].dropna()).union(set(_qdf["receiver"].dropna())))

    degree_count: dict = {}
    for _, row in _qdf.iterrows():
        degree_count[row["sender"]] = degree_count.get(row["sender"], 0) + 1
        degree_count[row["receiver"]] = degree_count.get(row["receiver"], 0) + 1

    degrees     = list(degree_count.values())
    avg_degree  = sum(degrees) / len(degrees) if degrees else 0
    max_degree  = max(degrees) if degrees else 0

    qc1, qc2, qc3, qc4 = st.columns(4)
    qc1.metric("Transactions",  f"{num_transactions:,}")
    qc2.metric("Accounts",      f"{num_nodes:,}")
    qc3.metric("Avg Degree",    round(avg_degree, 2))
    qc4.metric("Max Degree",    max_degree)

    # Quality warnings
    warnings: list = []
    if num_transactions < 5000:
        warnings.append("Dataset is too small for meaningful AML detection")
    if num_nodes < 1000:
        warnings.append("Low number of accounts — graph structure weak")
    if max_degree < 10:
        warnings.append("No strong hubs detected — structural signals weak")
    if avg_degree < 2:
        warnings.append("Very sparse graph — limited connectivity")

    if warnings:
        for w in warnings:
            st.warning(w)
    else:
        st.success("✅ Dataset structure looks suitable for AML analysis")

    # Degree distribution histogram
    with st.expander("📈 Degree Distribution", expanded=False):
        fig_deg = px.histogram(
            x=degrees,
            nbins=50,
            title="Node Degree Distribution",
            labels={"x": "Degree", "y": "Count"},
            template="plotly_dark",
            color_discrete_sequence=["#0099ff"],
        )
        fig_deg.update_layout(margin=dict(t=40, b=20, l=20, r=20))
        st.plotly_chart(fig_deg, use_container_width=True)
