"""
pages/2_Alerts.py — Alert Queue / Entry Point to Investigation
"""
import json
import pandas as pd
import streamlit as st

st.title("🚨 Alert Queue")
st.markdown("Click **Investigate** on any alert to open the investigation panel.")

# ── Load alerts ───────────────────────────────────────────────────────────────
def load_alerts():
    import os
    path = "data/alerts/alerts.json"
    if not os.path.exists(path):
        return []
    mtime = os.path.getmtime(path)
    return _load_alerts_cached(path, mtime)

@st.cache_data
def _load_alerts_cached(path, mtime):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

alerts = load_alerts()
df = pd.DataFrame(alerts).sort_values("risk_score", ascending=False).reset_index(drop=True)
df["rank"] = df.index + 1

# ── Risk tier badge ───────────────────────────────────────────────────────────
def risk_badge(score):
    if score >= 0.85:
        return "🔴 Critical"
    elif score >= 0.75:
        return "🟠 High"
    elif score >= 0.60:
        return "🟡 Medium"
    else:
        return "🟢 Low"

df["Risk Tier"] = df["risk_score"].apply(risk_badge)

# ── Filter controls ───────────────────────────────────────────────────────────
col_f1, col_f2 = st.columns([2, 1])
with col_f1:
    search = st.text_input("🔎 Search by Node ID", placeholder="e.g. ACC039705")
with col_f2:
    tier_filter = st.selectbox("Filter by Risk Tier", ["All", "🔴 Critical", "🟠 High", "🟡 Medium", "🟢 Low"])

filtered = df.copy()
if search:
    filtered = filtered[filtered["node_id"].str.contains(search, case=False)]
if tier_filter != "All":
    filtered = filtered[filtered["Risk Tier"] == tier_filter]

st.markdown(f"**{len(filtered)} alerts** displayed")
st.markdown("---")

# ── Alert rows ────────────────────────────────────────────────────────────────
ROWS_PER_PAGE = 25
total_pages = max(1, (len(filtered) - 1) // ROWS_PER_PAGE + 1)

if total_pages > 1:
    page_num = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
else:
    page_num = 1

start = (page_num - 1) * ROWS_PER_PAGE
page_df = filtered.iloc[start : start + ROWS_PER_PAGE]

# Header
h1, h2, h3, h4, h5 = st.columns([1, 3, 2, 2, 2])
h1.markdown("**#**")
h2.markdown("**Node ID**")
h3.markdown("**Risk Score**")
h4.markdown("**Risk Tier**")
h5.markdown("**Action**")
st.markdown("---")

for _, row in page_df.iterrows():
    c1, c2, c3, c4, c5 = st.columns([1, 3, 2, 2, 2])
    c1.write(row["rank"])
    c2.code(row["node_id"], language=None)
    c3.write(f"{row['risk_score']:.4f}")
    c4.write(row["Risk Tier"])
    if c5.button("🔬 Investigate", key=f"btn_{row['alert_id']}"):
        st.session_state["selected_node"] = row["node_id"]
        st.session_state["selected_case_id"] = None
        st.session_state["page"] = "🔬 Investigation"
        st.rerun()
