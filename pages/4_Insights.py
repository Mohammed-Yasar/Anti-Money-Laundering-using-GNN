"""
pages/4_Insights.py — Pattern Understanding
"""
import json
import pandas as pd
import plotly.express as px
import streamlit as st

st.title("💡 Insights")
st.markdown("Aggregate pattern analysis across all investigated cases.")

@st.cache_data
def load_cases():
    with open("data/cases/cases.json", encoding="utf-8") as f:
        return json.load(f)

cases    = load_cases()
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
    st.plotly_chart(fig1, use_container_width=True)

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
        st.plotly_chart(fig2, use_container_width=True)

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
        st.plotly_chart(fig3, use_container_width=True)

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
    st.plotly_chart(fig4, use_container_width=True)

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
    st.plotly_chart(fig5, use_container_width=True)
