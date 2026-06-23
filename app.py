"""
app.py — AML Investigation Dashboard
Entry point for the Streamlit multi-page application.
"""

import os
import time
import subprocess
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="AML Investigation Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar Navigation ──────────────────────────────────────────────────────
st.sidebar.title("🔍 AML Dashboard")
st.sidebar.markdown("---")

# ── Data Input ───────────────────────────────────────────────────────────────
st.sidebar.header("Data Input")

REQUIRED_COLUMNS = [
    "transaction_id",
    "sender",
    "receiver",
    "amount",
    "timestamp",
]

uploaded_file = st.sidebar.file_uploader(
    "Upload Transaction CSV",
    type=["csv"],
)

if uploaded_file is not None:
    try:
        _df = pd.read_csv(uploaded_file)
        missing_cols = [c for c in REQUIRED_COLUMNS if c not in _df.columns]
        if missing_cols:
            st.sidebar.error(f"Missing columns: {missing_cols}")
            st.stop()
        os.makedirs("data/raw", exist_ok=True)
        _df.to_csv("data/raw/uploaded_transactions.csv", index=False)
        st.sidebar.success(f"Uploaded {len(_df):,} rows — file saved.")
    except Exception as _e:
        st.sidebar.error(f"Could not read file: {_e}")
        st.stop()

if uploaded_file is not None:
    st.sidebar.info("🧠 Model: Pretrained on synthetic dataset\n\n⚙️ Running in Inference Mode")

if st.sidebar.button("▶ Run AML Pipeline"):
    is_inference = uploaded_file is not None
    env = os.environ.copy()
    if is_inference:
        env["AML_INFERENCE"] = "1"
        env["AML_DATA_PATH"] = "data/raw/uploaded_transactions.csv"
        spinner_msg = "Pipeline executing in Inference Mode... (No ground truth labels available — evaluation disabled)"
    else:
        spinner_msg = "Running AML pipeline… (this may take ~1–2 minutes)"
        
    with st.spinner(spinner_msg):
        import sys
        # Prefer sys.executable if it has torch, otherwise fall back to "python"
        python_exe = sys.executable
        try:
            import torch
        except ImportError:
            python_exe = "python"

        # Run GNN script
        res_gnn = subprocess.run(
            [python_exe, "main_gnn.py"],
            capture_output=True,
            text=True,
            env=env,
        )
        sys.stdout.write(res_gnn.stdout)
        sys.stderr.write(res_gnn.stderr)
        if res_gnn.returncode != 0:
            st.sidebar.error("Pipeline failed during GNN stage!")
            with st.expander("GNN Error Details", expanded=True):
                st.code(res_gnn.stderr)
            st.stop()
            
        # Run Investigation script
        res_inv = subprocess.run(
            [python_exe, "main_investigation.py"],
            capture_output=True,
            text=True,
            env=env,
        )
        sys.stdout.write(res_inv.stdout)
        sys.stderr.write(res_inv.stderr)
        if res_inv.returncode != 0:
            st.sidebar.error("Pipeline failed during Investigation stage!")
            with st.expander("Investigation Error Details", expanded=True):
                st.code(res_inv.stderr)
            st.stop()
            
    st.sidebar.success("Pipeline execution completed!")
    time.sleep(1)
    st.rerun()

if os.path.exists("data/raw/uploaded_transactions.csv"):
    if st.sidebar.button("🗑 Reset to Default Data", width="stretch"):
        os.remove("data/raw/uploaded_transactions.csv")
        st.sidebar.success("Resetting to original dataset...")
        time.sleep(1)
        st.rerun()

st.sidebar.markdown("---")

pages = {
    "📊 Overview":        "pages/1_Overview.py",
    "🚨 Alerts":          "pages/2_Alerts.py",
    "🔬 Investigation":   "pages/3_Investigation.py",
    "💡 Insights":        "pages/4_Insights.py",
}

# Initialise session state
if "page" not in st.session_state:
    st.session_state["page"] = "📊 Overview"
if "selected_node" not in st.session_state:
    st.session_state["selected_node"] = None
if "selected_case_id" not in st.session_state:
    st.session_state["selected_case_id"] = None

for label in pages:
    if st.sidebar.button(label):
        st.session_state["page"] = label

st.sidebar.caption("AML System · Analyst View")

# ── Route to selected page ───────────────────────────────────────────────────
current = st.session_state["page"]

if current == "📊 Overview":
    exec(open("pages/1_Overview.py", encoding="utf-8").read())
elif current == "🚨 Alerts":
    exec(open("pages/2_Alerts.py", encoding="utf-8").read())
elif current == "🔬 Investigation":
    exec(open("pages/3_Investigation.py", encoding="utf-8").read())
elif current == "💡 Insights":
    exec(open("pages/4_Insights.py", encoding="utf-8").read())
