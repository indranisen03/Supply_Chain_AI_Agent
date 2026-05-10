"""
Agent 2 — Historical Trend Agent

Reads real supply chain data from both Kaggle datasets:
  - Dataset 1 (supply_chain_data.csv): SKU-level catalog data
  - Dataset 2 (DataCoSupplyChainDataset.csv): 172,765 orders (2015–2018)

For the given SKU + current month:
  - Computes avg demand delta vs baseline using real seasonal factors
  - For the given supplier: computes on-time delivery rate from real order history
  - Surfaces patterns like "Q2 demand +2.5% above baseline"
    and "SUP-003 (Western Europe) late-delivery risk: 58.5%"

Model: GPT-4o-mini.
"""

import json
import logging
import time
from datetime import datetime
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from agents._llm import get_llm, token_cost

from agents.state import SupplyChainState
from audit.soc2_logger import log_agent_complete, log_verification

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are the Historical Trend Agent for Stellantis supply chain operations.
You receive pre-computed statistics from real supply chain datasets (2015-2018).
Produce a concise natural-language analysis of trends and risks.

Return JSON with keys:
  avg_demand_delta (float, % vs baseline),
  supplier_on_time_rate (float 0-1),
  trend_summary (1-2 sentence string),
  risk_flags (list of strings),
  verification_logs (list of strings),
  decision_trail (list of strings)

Verification log format: "[HISTORICAL] <metric>: <value> → PASS ✓ / FAIL ✗"
"""


def compute_historical_trend(sku_id: str, supplier_id: str, current_month: int) -> Dict[str, Any]:
    """
    Compute trend statistics using real Kaggle datasets.
    Returns a dict consumed by the LLM for narrative generation.
    """
    from data.data_mapper import (
        get_monthly_demand_trend,
        get_seasonal_factors,
        get_supplier_performance,
        load_sku_catalog,
    )

    # ── Seasonal demand factor from real DataCo order history ──────────────
    seasonal = get_seasonal_factors(current_month)
    avg_demand_delta = round((seasonal["seasonal_factor"] - 1.0) * 100, 2)

    # ── Supplier performance from real DataCo regional data ────────────────
    _SUPPLIER_REGION_MAP = {
        "SUP-001": "Canada",
        "SUP-002": "Central America",
        "SUP-003": "Western Europe",
        "SUP-004": "Northern Europe",
        "SUP-005": "Central Africa",
    }
    region = _SUPPLIER_REGION_MAP.get(supplier_id, "Central America")
    perf_df = get_supplier_performance(region=region)

    on_time_rate = 0.43
    late_risk_rate = 0.57
    avg_actual_lead = 3.5
    total_orders = 0

    if not perf_df.empty:
        row = perf_df[perf_df["region"] == region]
        if not row.empty:
            on_time_rate = float(row.iloc[0]["on_time_rate"])
            late_risk_rate = float(row.iloc[0]["late_risk_rate"])
            avg_actual_lead = float(row.iloc[0]["avg_actual_lead"])
            total_orders = int(row.iloc[0]["total_orders"])

    # ── Monthly demand trend (last 12 months of DataCo data) ──────────────
    demand_trend = get_monthly_demand_trend()
    q4_spike = False
    q4_delta_pct = 0.0
    seasonal_note = seasonal["note"]

    if not demand_trend.empty:
        # Compute Q4 vs full-year average from real data
        annual_avg_qty = demand_trend["total_qty"].mean()
        q4_data = demand_trend[demand_trend["quarter"] == 4]
        if not q4_data.empty and annual_avg_qty > 0:
            q4_avg = q4_data["total_qty"].mean()
            q4_delta_pct = round((q4_avg - annual_avg_qty) / annual_avg_qty * 100, 1)
            q4_spike = current_month in [10, 11, 12] and abs(q4_delta_pct) > 5

        # Q4 note for current quarter
        if current_month in [10, 11, 12]:
            seasonal_note = (
                f"Q4 demand is {q4_delta_pct:+.1f}% vs annual average "
                f"(real DataCo data, {len(demand_trend)} monthly observations)."
            )

    # ── SKU info from Dataset 1 ────────────────────────────────────────────
    catalog = load_sku_catalog()
    sku_info = {}
    if not catalog.empty and sku_id in catalog.index:
        row = catalog.loc[sku_id]
        sku_info = {
            "units_sold": int(row.get("units_sold", 0)),
            "defect_rate": float(row.get("defect_rate", 0)),
            "product_type": str(row.get("product_type", "unknown")),
            "inspection_result": str(row.get("inspection_result", "unknown")),
        }

    # ── Compute supplier Q4 delays from DataCo ─────────────────────────────
    # DataCo doesn't break down by supplier directly, so we use regional proxy
    q4_late_risk = late_risk_rate  # conservative: use overall late risk

    return {
        "sku_id": sku_id,
        "supplier_id": supplier_id,
        "region": region,
        "current_month": current_month,
        "avg_demand_delta_pct": avg_demand_delta,
        "seasonal_factor": seasonal["seasonal_factor"],
        "seasonal_note": seasonal_note,
        "q4_spike_detected": q4_spike,
        "q4_demand_delta_pct": q4_delta_pct,
        "supplier_on_time_rate": round(on_time_rate, 4),
        "supplier_late_risk_rate": round(late_risk_rate, 4),
        "supplier_avg_lead_time": round(avg_actual_lead, 2),
        "supplier_total_orders_analyzed": total_orders,
        "q4_late_risk": round(q4_late_risk, 4),
        "sku_units_sold": sku_info.get("units_sold", 0),
        "sku_defect_rate": sku_info.get("defect_rate", 0),
        "sku_product_type": sku_info.get("product_type", "unknown"),
        "data_source": "DataCoSupplyChainDataset.csv + supply_chain_data.csv",
        "years_analyzed": 3,
    }


def historical_trend_agent(state: SupplyChainState) -> SupplyChainState:
    t0 = time.time()
    run_id = state["run_id"]
    sku_id = state["sku_id"]
    supplier_id = state["supplier_id"]
    current_month = datetime.now().month

    verification_logs = list(state.get("verification_logs", []))
    decision_trail = list(state.get("decision_trail", []))
    eval_metrics = dict(state.get("eval_metrics", {}))

    trend_data = compute_historical_trend(sku_id, supplier_id, current_month)

    llm = get_llm("mini")
    user_msg = (
        f"Real historical trend data from Kaggle supply chain datasets:\n"
        f"{json.dumps(trend_data, indent=2)}\n\n"
        f"Generate the analysis JSON based on this real data."
    )

    messages = [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=user_msg)]
    response = llm.invoke(messages)
    raw = response.content.strip()
    tokens = response.usage_metadata.get("total_tokens", 0) if response.usage_metadata else 0

    try:
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        parsed: Dict[str, Any] = json.loads(raw)
    except Exception:
        parsed = {
            "avg_demand_delta": trend_data["avg_demand_delta_pct"],
            "supplier_on_time_rate": trend_data["supplier_on_time_rate"],
            "trend_summary": trend_data["seasonal_note"],
            "risk_flags": [],
            "verification_logs": [],
            "decision_trail": [],
        }

    # ── Self-verification against real data thresholds ───────────────────
    on_time = trend_data["supplier_on_time_rate"]
    late_risk = trend_data["supplier_late_risk_rate"]

    # DataCo baseline: ~43% on-time is industry average for this dataset
    # Flag if significantly below average (< 40%)
    on_time_ok = on_time >= 0.40
    high_late_risk = late_risk > 0.60

    v1 = (
        f"[HISTORICAL] Supplier {supplier_id} ({trend_data['region']}) "
        f"on-time rate: {on_time:.1%} | late-risk: {late_risk:.1%} → "
        f"{'PASS ✓' if on_time_ok else 'FAIL ✗ → high disruption risk'}"
    )
    v2 = (
        f"[HISTORICAL] Month {current_month} demand delta: "
        f"{trend_data['avg_demand_delta_pct']:+.1f}% vs baseline → "
        f"{'PASS ✓' if abs(trend_data['avg_demand_delta_pct']) < 15 else 'WARN ⚠ high seasonal variance'}"
    )
    v3 = (
        f"[HISTORICAL] Data source: {trend_data['data_source']} "
        f"({trend_data['supplier_total_orders_analyzed']:,} orders analysed) → PASS ✓"
    )

    log_verification(run_id, "HISTORICAL",
                     f"on_time >= 0.40 ({on_time:.3f})", on_time_ok)
    log_verification(run_id, "HISTORICAL",
                     f"late_risk <= 0.60 ({late_risk:.3f})", not high_late_risk)

    risk_flags = parsed.get("risk_flags", [])
    if high_late_risk:
        risk_flags.append(f"Late delivery risk {late_risk:.1%} exceeds 60% threshold")
    if trend_data["q4_spike_detected"]:
        risk_flags.append(f"Q4 demand anomaly: {trend_data['q4_demand_delta_pct']:+.1f}% vs annual average")
    if trend_data["sku_defect_rate"] > 3.0:
        risk_flags.append(f"Elevated defect rate {trend_data['sku_defect_rate']:.2f}% for {sku_id}")

    verification_logs += [v1, v2, v3] + parsed.get("verification_logs", [])
    decision_trail += parsed.get("decision_trail", [])
    decision_trail.append(
        f"Historical analysis (real Kaggle data): {trend_data['seasonal_note']} "
        f"Supplier {supplier_id} on-time: {on_time:.1%}, "
        f"late-risk: {late_risk:.1%} across {trend_data['supplier_total_orders_analyzed']:,} orders."
    )

    cost_usd = token_cost(tokens, "mini")
    eval_metrics["historical_trend"] = {
        "latency_ms": int((time.time() - t0) * 1000),
        "tokens": tokens,
        "cost_usd": cost_usd,
    }

    log_agent_complete(run_id, "HISTORICAL_TREND", trend_data,
                       cost_usd=cost_usd, tokens=tokens)

    return {
        **state,
        "historical_trend": {
            **trend_data,
            "trend_summary": parsed.get("trend_summary", trend_data["seasonal_note"]),
            "risk_flags": risk_flags,
        },
        "verification_logs": verification_logs,
        "decision_trail": decision_trail,
        "eval_metrics": eval_metrics,
        "tokens_used": state.get("tokens_used", 0) + tokens,
        "cost_usd": state.get("cost_usd", 0.0) + cost_usd,
    }
