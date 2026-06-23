"""
pages/3_Investigation.py — Core Investigation Page
Graph visualization (70%) + Case panel (30%)
"""
import json
import sys
import os
import networkx as nx
import plotly.graph_objects as go
import streamlit as st

# ── Load cases ────────────────────────────────────────────────────────────────
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

cases = load_cases()
case_map      = {c["node_id"]:  c for c in cases}
case_id_map   = {c["case_id"]: c for c in cases}

# ── Plotly graph renderer (purely from JSON payload) ─────────────────────────
def plot_subgraph_from_case(case_graph, central_node):
    if not case_graph or not case_graph.get("nodes"):
        return None
        
    G_sub = nx.DiGraph()
    G_sub.add_nodes_from(case_graph["nodes"])
    for e in case_graph["edges"]:
        G_sub.add_edge(e["source"], e["target"])
        
    pos = nx.spring_layout(G_sub, seed=42, k=1.5)

    # Edges
    edge_x, edge_y = [], []
    for u, v in G_sub.edges():
        if u in pos and v in pos:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=1, color="#444"),
        hoverinfo="none",
    )

    # Nodes
    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    for node in G_sub.nodes():
        if node in pos:
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            if node == central_node:
                node_color.append("#e74c3c")
                node_size.append(20)
            else:
                node_color.append("#3498db")
                node_size.append(10)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        textfont=dict(size=8, color="#ccc"),
        hoverinfo="text",
        marker=dict(
            color=node_color,
            size=node_size,
            line=dict(width=1, color="#111"),
        ),
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            template="plotly_dark",
            paper_bgcolor="#0e0e1a",
            plot_bgcolor="#0e0e1a",
            showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=550,
        ),
    )
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# PAGE LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
st.title("🔬 Investigation")

# ── Node selector (manual + session_state) ────────────────────────────────────
node_ids_with_cases = list(case_map.keys())

preset = st.session_state.get("selected_node")

# If a similar case was clicked, resolve its node
selected_case_id_from_session = st.session_state.get("selected_case_id")
if selected_case_id_from_session and selected_case_id_from_session in case_id_map:
    preset = case_id_map[selected_case_id_from_session]["node_id"]
    st.session_state["selected_node"] = preset

default_idx = 0
if preset in node_ids_with_cases:
    default_idx = node_ids_with_cases.index(preset)

selected_node = st.selectbox(
    "Select Account / Node",
    options=node_ids_with_cases,
    index=default_idx,
    key="inv_node_select",
)
st.session_state["selected_node"] = selected_node

# ── Main layout: 70/30 split ──────────────────────────────────────────────────
graph_col, panel_col = st.columns([7, 3])

case = case_map.get(selected_node)

with graph_col:
    st.subheader(f"Transaction Subgraph — {selected_node}")
    
    if not case:
        st.info("No case found for this node.")
    else:
        case_graph = case.get("graph")
        if case_graph and case_graph.get("nodes"):
            fig = plot_subgraph_from_case(case_graph, selected_node)
            if fig:
                st.plotly_chart(fig)
                st.caption(
                    f"🔴 Central node · 🔵 Neighbours · "
                    f"**{case.get('subgraph_nodes', 0)}** nodes · **{case.get('subgraph_edges', 0)}** edges"
                )
            else:
                st.warning("Could not render subgraph despite graph payload.")
        else:
            st.info("Visualisation payload pending from investigation subsystem.")

with panel_col:
    if not case:
        st.info("No case details to display.")
    else:
        # ── Case header ──────────────────────────────────────────────────────
        st.subheader("📋 Case Details")

        risk = case.get("risk_score", 0)
        risk_color = "#e74c3c" if risk >= 0.85 else "#e67e22" if risk >= 0.75 else "#f1c40f"

        st.markdown(
            f"""
            <div style="background:#1a1a2e;border-radius:8px;padding:14px 16px;margin-bottom:12px;">
                <table style="width:100%;color:#ddd;font-size:13px;border-collapse:collapse;">
                    <tr><td style="padding:4px 0;color:#888;">Case ID</td>
                        <td style="font-weight:600;">{case.get('case_id','—')}</td></tr>
                    <tr><td style="padding:4px 0;color:#888;">Node ID</td>
                        <td style="font-weight:600;">{case.get('node_id','—')}</td></tr>
                    <tr><td style="padding:4px 0;color:#888;">Risk Score</td>
                        <td style="font-weight:700;color:{risk_color};">{risk:.4f}</td></tr>
                    <tr><td style="padding:4px 0;color:#888;">Typology</td>
                        <td style="font-weight:600;text-transform:capitalize;">{case.get('typology','—')}</td></tr>
                    <tr><td style="padding:4px 0;color:#888;">Subgraph</td>
                        <td>{case.get('subgraph_nodes','—')} nodes · {case.get('subgraph_edges','—')} edges</td></tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Motifs ───────────────────────────────────────────────────────────
        st.markdown("**🔎 Motifs Detected**")
        motifs = case.get("motifs_detected", {})
        m_cols = st.columns(3)
        for i, (m_name, detected) in enumerate(motifs.items()):
            icon   = "✅" if detected else "❌"
            color  = "#2ecc71" if detected else "#555"
            m_cols[i].markdown(
                f"<div style='text-align:center;color:{color};font-size:12px;'>"
                f"{icon}<br>{m_name.capitalize()}</div>",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Temporal signals ─────────────────────────────────────────────────
        st.markdown("**⏱️ Temporal Signals**")
        temporal = case.get("temporal_anomalies", {})
        t_cols = st.columns(2)
        for i, (t_name, active) in enumerate(temporal.items()):
            icon  = "✅" if active else "❌"
            color = "#e74c3c" if active else "#555"
            t_cols[i].markdown(
                f"<div style='text-align:center;color:{color};font-size:12px;'>"
                f"{icon}<br>{t_name.replace('_',' ').capitalize()}</div>",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Explanation ──────────────────────────────────────────────────────
        st.markdown("**📝 Explanation**")
        explanations = case.get("explanation", [])
        if explanations:
            for exp in explanations:
                st.markdown(f"- {exp}")
        else:
            st.caption("No explanation available.")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Similar Cases ────────────────────────────────────────────────────
        st.markdown("**🔗 Similar Cases**")
        similar = case.get("similar_cases", [])
        if similar:
            for sc_id in similar:
                if st.button(f"📂 {sc_id}", key=f"sim_{sc_id}_{selected_node}"):
                    if sc_id in case_id_map:
                        st.session_state["selected_case_id"] = sc_id
                        st.session_state["selected_node"] = case_id_map[sc_id]["node_id"]
                        st.rerun()
        else:
            st.caption("No similar cases found.")
