"""
pages/4_Insights.py — Pattern Understanding
"""
import json
import pandas as pd
import plotly.express as px
import streamlit as st

st.title("💡 Insights")
st.markdown("Aggregate pattern analysis across all investigated cases.")

def load_cases():
    import os
    path = "data/cases/cases.json"
    if not os.path.exists(path):
        return []
    mtime = os.path.getmtime(path)
    return _load_cases_cached(path, mtime)

@st.cache_data
def _load_cases_cached(path, mtime):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

cases    = load_cases()
if not cases:
    st.info("ℹ️ **No cases available**: Insights will be available after you run the **AML Pipeline** to process investigations.")
    st.stop()
cases_df = pd.DataFrame(cases)

# ── Typology Distribution ─────────────────────────────────────────────────────
st.subheader("Typology Distribution")
if "typology" in cases_df.columns:
    typo_counts = cases_df["typology"].value_counts().reset_index()
    typo_counts.columns = ["Typology", "Count"]
    fig1 = px.bar(
        typo_counts, x="Typology", y="Count",
        color="Typology",
        text_auto=True,
        color_discrete_sequence=px.colors.qualitative.Bold,
        template="plotly_dark",
        title="Cases by Typology",
    )
    fig1.update_layout(showlegend=False, margin=dict(t=40))
    st.plotly_chart(fig1, width="stretch")

st.markdown("---")

# ── Risk Score Histogram ──────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Risk Score Distribution")
    if "risk_score" in cases_df.columns:
        fig2 = px.histogram(
            cases_df, x="risk_score",
            nbins=40,
            color_discrete_sequence=["#e74c3c"],
            template="plotly_dark",
            title="Risk Score Histogram",
            labels={"risk_score": "Risk Score"},
        )
        fig2.update_layout(margin=dict(t=40))
        st.plotly_chart(fig2, width="stretch")

with col2:
    st.subheader("Subgraph Size Distribution")
    if "subgraph_nodes" in cases_df.columns:
        fig3 = px.histogram(
            cases_df, x="subgraph_nodes",
            nbins=30,
            color_discrete_sequence=["#3498db"],
            template="plotly_dark",
            title="Subgraph Node Count",
            labels={"subgraph_nodes": "Nodes in Subgraph"},
        )
        fig3.update_layout(margin=dict(t=40))
        st.plotly_chart(fig3, width="stretch")

st.markdown("---")

# ── Motif co-occurrence table ─────────────────────────────────────────────────
st.subheader("Motif Prevalence")
if "motifs_detected" in cases_df.columns:
    motif_df = pd.json_normalize(cases_df["motifs_detected"])
    motif_pct = (motif_df.sum() / len(motif_df) * 100).round(1).reset_index()
    motif_pct.columns = ["Motif", "% of Cases"]
    fig4 = px.bar(
        motif_pct, x="Motif", y="% of Cases",
        color="Motif",
        text_auto=True,
        color_discrete_sequence=["#9b59b6", "#2ecc71", "#e67e22"],
        template="plotly_dark",
        title="Motif Detection Rate (%)",
    )
    fig4.update_layout(showlegend=False, margin=dict(t=40))
    st.plotly_chart(fig4, width="stretch")

# ── Risk score by typology (box plot) ────────────────────────────────────────
st.subheader("Risk Score by Typology")
if "typology" in cases_df.columns and "risk_score" in cases_df.columns:
    fig5 = px.box(
        cases_df, x="typology", y="risk_score",
        color="typology",
        color_discrete_sequence=px.colors.qualitative.Bold,
        template="plotly_dark",
        title="Risk Score Distribution by Typology",
        labels={"typology": "Typology", "risk_score": "Risk Score"},
    )
    fig5.update_layout(showlegend=False, margin=dict(t=40))
    st.plotly_chart(fig5, width="stretch")
