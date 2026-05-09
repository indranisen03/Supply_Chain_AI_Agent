"""
Agent 6 — Executive Summarizer

Reads full shared state and outputs a 5-6 sentence plain-English summary
covering: what was detected, historical context, simulation outcome,
recommendation, HITL tier, and judge verdict.
Model: GPT-4o-mini (simple task, save cost).
"""

import json
import time
import logging
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agents.state import SupplyChainState
from audit.soc2_logger import log_agent_complete
from config import MODEL_MINI, OPENAI_API_KEY

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are the Executive Summarizer for Stellantis supply chain AI.
Write a concise 5-6 sentence plain English summary for a business audience.

Cover ALL of these points:
1. What supply chain condition was detected (inventory vs safety stock, disruptions)
2. Historical context (seasonal trends, supplier performance over 3 years)
3. Simulation outcome (stockout risk scenarios, decision window)
4. Purchase order recommendation (qty, supplier, value, required-by date)
5. HITL tier and what it means for the procurement team
6. LLM Judge verdict and confidence

Write in professional business tone. No bullet points — flowing paragraph.
Return plain text, NOT JSON.
"""


def summarizer_agent(state: SupplyChainState) -> SupplyChainState:
    t0 = time.time()
    run_id = state["run_id"]

    verification_logs = list(state.get("verification_logs", []))
    decision_trail = list(state.get("decision_trail", []))
    eval_metrics = dict(state.get("eval_metrics", {}))

    po = state.get("po_recommendation", {})
    trend = state.get("historical_trend", {})

    user_msg = f"""
Supply chain run summary data:

SKU: {state['sku_id']} | Warehouse: {state['warehouse']} | Supplier: {state['supplier_id']}
Inventory: {state.get('inventory_level', 'N/A')} units (safety stock: {state.get('safety_stock_threshold', 'N/A')})
Disruption: {state.get('disruption_detected', False)} — {state.get('disruption_details', {}).get('details', 'none')}
Historical trend: {trend.get('seasonal_note', 'N/A')} | Supplier on-time: {trend.get('supplier_on_time_rate', 'N/A'):.1%}
Demand forecast (14d): {state.get('demand_forecast', 'N/A')} units | Lead time: {state.get('lead_time_days', 'N/A')} days
Decision window: {state.get('decision_window_days', 'N/A')} days until stockout risk >50%

PO Recommendation:
  Quantity: {po.get('quantity', 'N/A')} units | Supplier: {po.get('supplier_id', 'N/A')}
  Required by: {po.get('required_by', 'N/A')} | Value: ${po.get('estimated_value', 0):,.2f}
  Confidence: {po.get('confidence', 0):.1%}

HITL Tier: {state.get('hitl_tier', 'N/A')} | Human approved: {state.get('human_approved', False)}
Judge verdict: {state.get('judge_verdict', 'N/A')} | Score: {state.get('judge_score', 0):.1f}/10
Judge flags: {state.get('judge_flags', [])}

Write the executive summary now.
"""

    llm = ChatOpenAI(model=MODEL_MINI, api_key=OPENAI_API_KEY, temperature=0.3, max_tokens=300)
    messages = [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=user_msg)]

    try:
        response = llm.invoke(messages)
        summary = response.content.strip()
        tokens = response.usage_metadata.get("total_tokens", 0) if response.usage_metadata else 0
    except Exception as exc:
        logger.error("Summarizer LLM call failed: %s", exc)
        summary = (
            f"Supply chain monitoring detected {state['sku_id']} inventory at "
            f"{state.get('inventory_level', 0)} units, below the safety stock of "
            f"{state.get('safety_stock_threshold', 0)}. "
            f"A purchase order for {po.get('quantity', 0)} units has been recommended "
            f"with HITL tier {state.get('hitl_tier', 'N/A')}. "
            f"The LLM judge scored {state.get('judge_score', 0):.1f}/10 with verdict "
            f"{state.get('judge_verdict', 'N/A')}."
        )
        tokens = 0

    decision_trail.append(f"Executive summary generated.")

    cost_usd = (tokens / 1_000_000) * 0.60
    eval_metrics["summarizer"] = {"latency_ms": int((time.time() - t0) * 1000), "tokens": tokens, "cost_usd": cost_usd}

    log_agent_complete(run_id, "SUMMARIZER", {"summary": summary[:200]}, cost_usd=cost_usd, tokens=tokens)

    return {
        **state,
        "executive_summary": summary,
        "decision_trail": decision_trail,
        "eval_metrics": eval_metrics,
        "tokens_used": state.get("tokens_used", 0) + tokens,
        "cost_usd": state.get("cost_usd", 0.0) + cost_usd,
    }
