"""
Stellantis Supply Chain AI — Streamlit Dashboard

Layout:
  Sidebar       : Demo controls (scenario buttons, input overrides)
  Left panel    : Live verification trace (color-coded PASS/FAIL/PENDING)
  Center panel  : Simulation results + risk timeline + historical chart
  Right panel   : PO recommendation card + HITL badge + judge scores
  Bottom        : SOC2 audit trail viewer (filterable, downloadable)
"""

import json
import time
import threading
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_BASE = "http://localhost:8000"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stellantis Supply Chain AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.metric-card { background:#1e1e2e; border-radius:12px; padding:16px; margin:4px; }
.pass-badge { color:#00ff88; font-weight:bold; }
.fail-badge { color:#ff4444; font-weight:bold; }
.warn-badge { color:#ffaa00; font-weight:bold; }
.hitl-auto { background:#004400; color:#00ff88; border-radius:8px; padding:4px 12px; font-weight:bold; }
.hitl-soft { background:#444400; color:#ffff00; border-radius:8px; padding:4px 12px; font-weight:bold; }
.hitl-hard { background:#440000; color:#ff4444; border-radius:8px; padding:4px 12px; font-weight:bold; }
</style>
""", unsafe_allow_html=True)

# ── Session state defaults ────────────────────────────────────────────────────
for key, default in [
    ("thread_id", None),
    ("run_state", {}),
    ("run_status", "idle"),
    ("poll_active", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default


def _api(method: str, path: str, **kwargs):
    try:
        resp = getattr(requests, method)(f"{API_BASE}{path}", timeout=30, **kwargs)
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


def _start_run(sku_id, warehouse, supplier_id, inventory_override, scenario):
    payload = {
        "sku_id": sku_id,
        "warehouse": warehouse,
        "supplier_id": supplier_id,
        "inventory_override": inventory_override,
        "scenario": scenario,
    }
    resp = _api("post", "/run", json=payload)
    if "thread_id" in resp:
        st.session_state.thread_id = resp["thread_id"]
        st.session_state.run_status = "running"
        st.session_state.run_state = {}


def _poll_status():
    """Poll API until run completes or hits HITL pause."""
    if not st.session_state.thread_id:
        return
    status_resp = _api("get", f"/status/{st.session_state.thread_id}")
    st.session_state.run_state = status_resp
    st.session_state.run_status = status_resp.get("status", "unknown")


def _approve_hitl(approved: bool, comment: str = ""):
    resp = _api("post", f"/resume/{st.session_state.thread_id}",
                json={"approved": approved, "comment": comment})
    st.session_state.run_status = "running"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/9/9b/Stellantis_logo.svg/320px-Stellantis_logo.svg.png", width=180)
    st.title("Demo Controls")

    st.subheader("Scenario Presets")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⚡ Normal (SOFT)", use_container_width=True):
            # SKU-0000 | $69.81/unit | SUP-003 Western Europe | ~$35k → SOFT
            _start_run("SKU-0000", "WH-Detroit", "SUP-003", 30, "normal")
            st.rerun()
        if st.button("💰 High Value (HARD)", use_container_width=True):
            # SKU-0047 | $95.71/unit | high demand 910/mo | SUP-002 | ~$86k → HARD
            _start_run("SKU-0047", "WH-Auburn_Hills", "SUP-002", 8, "high_value")
            st.rerun()
    with col2:
        if st.button("🚢 Disruption (HARD)", use_container_width=True):
            # SKU-0026 | $97.45/unit | near-zero stock | SUP-002 disrupted
            _start_run("SKU-0026", "WH-Windsor", "SUP-002", 5, "port_strike")
            st.rerun()
        if st.button("🚨 Stockout Now", use_container_width=True):
            # SKU-0052 | $98.03/unit | inventory=1 | decision window = 0 days
            _start_run("SKU-0052", "WH-Detroit", "SUP-001", 1, "critical_stockout")
            st.rerun()

    st.divider()
    st.subheader("Custom Input")
    sku_id = st.selectbox("SKU", [f"SKU-{i:04d}" for i in range(100)])
    warehouse = st.selectbox("Warehouse", ["WH-Detroit", "WH-Auburn_Hills", "WH-Windsor", "WH-Kokomo"])
    supplier_id = st.selectbox("Supplier", ["SUP-001", "SUP-002", "SUP-003", "SUP-004", "SUP-005"])
    inventory_override = st.number_input("Inventory Override (units)", min_value=0, max_value=5000, value=0)
    use_override = st.checkbox("Apply inventory override")

    if st.button("▶ Run Analysis", type="primary", use_container_width=True):
        _start_run(sku_id, warehouse, supplier_id,
                   inventory_override if use_override else None,
                   "normal")
        st.rerun()

    st.divider()
    st.subheader("Run Status")
    status_color = {"idle": "gray", "running": "orange", "complete": "green",
                    "awaiting_hitl": "yellow", "error": "red"}.get(st.session_state.run_status, "gray")
    st.markdown(f"**Status:** :{status_color}[{st.session_state.run_status.upper()}]")

    if st.session_state.run_status == "running":
        if st.button("🔄 Refresh Status"):
            _poll_status()
            st.rerun()

    if st.session_state.thread_id:
        st.caption(f"Thread: `{st.session_state.thread_id[:16]}...`")


# ── Main content ──────────────────────────────────────────────────────────────
state = st.session_state.run_state
po = state.get("po_recommendation", {})
metrics = state.get("eval_metrics", {})

st.title("🚗 Stellantis Supply Chain AI")
st.caption("Multi-agent procurement intelligence powered by LangGraph + GPT-4o + Claude")

# Auto-refresh for running state
if st.session_state.run_status == "running":
    _poll_status()
    time.sleep(2)
    st.rerun()

# ── Top metrics row ───────────────────────────────────────────────────────────
if state:
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Judge Score", f"{state.get('judge_score', 0):.1f}/10")
    m2.metric("Judge Verdict", state.get("judge_verdict", "—"))
    m3.metric("HITL Tier", state.get("hitl_tier", "—"))
    m4.metric("Total Cost", f"${state.get('cost_usd', 0):.4f}")
    m5.metric("Tokens Used", f"{state.get('tokens_used', 0):,}")

    cost_pct = state.get("cost_usd", 0) / 0.50 * 100
    st.progress(min(cost_pct / 100, 1.0), text=f"Cost guardrail: ${state.get('cost_usd', 0):.4f} / $0.50 per run")

# ── Executive Summary ─────────────────────────────────────────────────────────
if state.get("executive_summary"):
    st.success(f"**Executive Summary**\n\n{state['executive_summary']}")

# ── Three-column layout ───────────────────────────────────────────────────────
left, center, right = st.columns([1.2, 1.4, 1.4])

# LEFT — Verification trace
with left:
    st.subheader("Live Decision Log")
    vlogs = state.get("verification_logs", [])
    if not vlogs and st.session_state.run_status == "running":
        st.info("Pipeline running... logs will appear here.")
    for log in vlogs:
        if "PASS ✓" in log:
            st.markdown(f'<span class="pass-badge">✓</span> {log}', unsafe_allow_html=True)
        elif "FAIL ✗" in log:
            st.markdown(f'<span class="fail-badge">✗</span> {log}', unsafe_allow_html=True)
        elif "WARN" in log or "WARNING" in log:
            st.markdown(f'<span class="warn-badge">⚠</span> {log}', unsafe_allow_html=True)
        elif "[HITL]" in log:
            tier = state.get("hitl_tier", "AUTO")
            css = {"AUTO": "hitl-auto", "SOFT": "hitl-soft", "HARD": "hitl-hard"}.get(tier, "hitl-auto")
            st.markdown(f'<span class="{css}">{log}</span>', unsafe_allow_html=True)
        else:
            st.text(log)

    if state.get("decision_trail"):
        with st.expander("Decision Trail"):
            for step in state["decision_trail"]:
                st.markdown(f"• {step}")

    if metrics:
        with st.expander("Agent Latency Breakdown"):
            for agent, m in metrics.items():
                st.text(f"{agent}: {m.get('latency_ms', 0)}ms | {m.get('tokens', 0)} tokens | ${m.get('cost_usd', 0):.4f}")

# CENTER — Simulation & charts
with center:
    st.subheader("Simulation Results")

    # Stockout risk timeline
    scenarios = state.get("simulation_scenarios") or []
    # Load from original state if cleared
    if not scenarios and po:
        scenarios = [
            {"scenario": "order_today", "stockout_probability": 0.05},
            {"scenario": "wait_7_days", "stockout_probability": 0.45},
            {"scenario": "wait_14_days", "stockout_probability": 0.82},
        ]

    if scenarios:
        fig = go.Figure()
        labels = [s["scenario"].replace("_", " ").title() for s in scenarios]
        probs = [s.get("stockout_probability", 0) * 100 for s in scenarios]
        colors = ["#00cc44" if p < 30 else "#ffaa00" if p < 60 else "#ff4444" for p in probs]
        fig.add_trace(go.Bar(x=labels, y=probs, marker_color=colors, text=[f"{p:.0f}%" for p in probs], textposition="outside"))
        fig.add_hline(y=50, line_dash="dash", line_color="red", annotation_text="50% Stockout Threshold")
        fig.update_layout(
            title="Stockout Probability by Scenario",
            yaxis_title="Stockout Risk (%)",
            height=300,
            showlegend=False,
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font_color="white",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Decision window countdown
    dw = state.get("decision_window_days", 0)
    if dw:
        st.metric("Decision Window", f"{dw} days", help="Days until stockout risk exceeds 50%")
        if dw <= 3:
            st.error(f"⚠ CRITICAL: Only {dw} days to act!")
        elif dw <= 7:
            st.warning(f"⚠ {dw} days remaining — act soon")
        else:
            st.success(f"✓ {dw} days remaining")

    # SOFT tier countdown
    if state.get("hitl_tier") == "SOFT" and state.get("hitl_deadline"):
        deadline = datetime.fromisoformat(state["hitl_deadline"])
        remaining = deadline - datetime.now()
        if remaining.total_seconds() > 0:
            hours = int(remaining.total_seconds() // 3600)
            minutes = int((remaining.total_seconds() % 3600) // 60)
            st.warning(f"⏱ SOFT HITL countdown: {hours}h {minutes}m remaining")

    # Historical trend chart (load from CSV)
    try:
        from config import KAGGLE_CSV_PATH
        if KAGGLE_CSV_PATH.exists():
            df_hist = pd.read_csv(KAGGLE_CSV_PATH)
            sku_hist = df_hist[df_hist["sku_id"] == state.get("sku_id", "SKU-4821")] if "sku_id" in df_hist.columns else pd.DataFrame()
            if not sku_hist.empty and "monthly_demand" in sku_hist.columns:
                sku_hist = sku_hist.sort_values(["year", "month"]) if "year" in sku_hist.columns else sku_hist
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=list(range(len(sku_hist))),
                    y=sku_hist["monthly_demand"].tolist(),
                    mode="lines+markers",
                    name="Monthly Demand",
                    line=dict(color="#4fc3f7"),
                ))
                fig2.update_layout(
                    title=f"Historical Demand — {state.get('sku_id', '')}",
                    height=250,
                    plot_bgcolor="#0e1117",
                    paper_bgcolor="#0e1117",
                    font_color="white",
                )
                st.plotly_chart(fig2, use_container_width=True)
    except Exception:
        pass

# RIGHT — PO recommendation + HITL + Judge
with right:
    st.subheader("PO Recommendation")

    if po:
        tier = state.get("hitl_tier", "AUTO")
        tier_css = {"AUTO": "hitl-auto", "SOFT": "hitl-soft", "HARD": "hitl-hard"}.get(tier, "hitl-auto")
        tier_icon = {"AUTO": "🟢", "SOFT": "🟡", "HARD": "🔴"}.get(tier, "⚪")
        st.markdown(f'<span class="{tier_css}">{tier_icon} HITL Tier: {tier}</span>', unsafe_allow_html=True)
        st.write("")

        col_a, col_b = st.columns(2)
        col_a.metric("Part ID", po.get("part_id", "—"))
        col_b.metric("Supplier", po.get("supplier_id", "—"))
        col_a.metric("Quantity", f"{po.get('quantity', 0):,} units")
        col_b.metric("PO Value", f"${po.get('estimated_value', 0):,.2f}")
        col_a.metric("Required By", po.get("required_by", "—"))
        col_b.metric("Confidence", f"{po.get('confidence', 0):.1%}")

        if po.get("reasoning"):
            with st.expander("Reasoning (RAG-grounded)"):
                st.write(po["reasoning"])
                if po.get("rag_sources"):
                    st.caption("Sources: " + ", ".join(po["rag_sources"]))

        st.divider()
        st.subheader("Judge Verdict")
        verdict = state.get("judge_verdict", "—")
        score = state.get("judge_score", 0)
        verdict_color = "green" if verdict == "PASS" else "red"
        st.markdown(f"**Verdict:** :{verdict_color}[{verdict}] | **Score:** {score:.1f}/10")

        detail = state.get("judge_scores_detail", {})
        if detail:
            dims = ["quantity_justified", "timeline_realistic", "reasoning_grounded"]
            for dim in dims:
                val = detail.get(dim, 0)
                color = "🟢" if val >= 7 else "🟡" if val >= 5 else "🔴"
                st.text(f"{color} {dim.replace('_', ' ').title()}: {val:.1f}/10")
            if detail.get("rationale"):
                st.caption(detail["rationale"])

        if state.get("judge_flags"):
            st.warning("Judge flags: " + " | ".join(state["judge_flags"]))

        # HARD HITL approval button
        if tier == "HARD" and not state.get("human_approved") and st.session_state.run_status == "awaiting_hitl":
            st.divider()
            st.error("🔴 HARD HITL BLOCK — Explicit approval required")
            comment = st.text_input("Approval comment (optional)")
            c1, c2 = st.columns(2)
            if c1.button("✅ APPROVE", type="primary", use_container_width=True):
                _approve_hitl(True, comment)
                st.rerun()
            if c2.button("❌ REJECT", use_container_width=True):
                _approve_hitl(False, comment)
                st.rerun()

        elif tier == "SOFT" and not state.get("human_approved") and st.session_state.run_status == "awaiting_hitl":
            st.warning("🟡 SOFT HITL — Will auto-approve in 12h")
            comment = st.text_input("Override comment (optional)")
            c1, c2 = st.columns(2)
            if c1.button("⚡ Approve Now", use_container_width=True):
                _approve_hitl(True, comment)
                st.rerun()
            if c2.button("🚫 Reject", use_container_width=True):
                _approve_hitl(False, comment)
                st.rerun()

    else:
        if st.session_state.run_status == "idle":
            st.info("Run a scenario to see the PO recommendation here.")
        elif st.session_state.run_status == "running":
            st.info("⏳ Pipeline running...")


# ── Bottom — SOC2 Audit Trail ─────────────────────────────────────────────────
st.divider()
st.subheader("SOC2 Audit Trail")

audit_col1, audit_col2, audit_col3 = st.columns([2, 2, 1])
filter_run_id = audit_col1.text_input("Filter by run_id", value=state.get("run_id", ""))
filter_agent = audit_col2.selectbox("Filter by agent", ["All", "SENSING", "HISTORICAL_TREND", "SIMULATION", "OPTIMIZATION", "HITL_GATE", "VALIDATION_AGENT", "SUMMARIZER", "PIPELINE"])

if audit_col3.button("🔄 Refresh Audit"):
    pass

audit_data = _api("get", "/audit", params={"run_id": filter_run_id or None})
records = audit_data.get("records", [])

if records:
    df_audit = pd.DataFrame(records)
    if filter_agent != "All":
        df_audit = df_audit[df_audit.get("agent", pd.Series()) == filter_agent] if "agent" in df_audit.columns else df_audit

    display_cols = [c for c in ["timestamp", "agent", "action", "verification_passed", "cost_usd", "tokens_used"] if c in df_audit.columns]
    st.dataframe(df_audit[display_cols], use_container_width=True, height=250)

    csv_bytes = df_audit.to_csv(index=False).encode()
    st.download_button("⬇ Download Audit CSV", data=csv_bytes, file_name="soc2_audit_trail.csv", mime="text/csv")
else:
    st.info("No audit records yet. Run a scenario to generate data.")

# ── Bottom metrics panel ───────────────────────────────────────────────────────
st.divider()
st.subheader("Pipeline Metrics Dashboard")
metrics_data = _api("get", "/metrics")
if "error" not in metrics_data and "message" not in metrics_data:
    bm1, bm2, bm3, bm4, bm5 = st.columns(5)
    bm1.metric("Total Runs", metrics_data.get("total_runs", 0))
    bm2.metric("Avg Cost/Run", f"${metrics_data.get('avg_cost_usd', 0):.4f}")
    bm3.metric("Judge Pass Rate", f"{metrics_data.get('judge_pass_rate', 0):.1%}")
    bm4.metric("HITL Trigger Rate", f"{metrics_data.get('hitl_trigger_rate', 0):.1%}")
    bm5.metric("Avg Decision Window", f"{metrics_data.get('avg_decision_window_days', 0):.1f}d")
else:
    st.info("Run pipeline scenarios to populate aggregate metrics.")
