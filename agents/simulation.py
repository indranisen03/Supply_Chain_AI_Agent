"""
Agent 3 — Simulation Agent

Calls demand forecast + lead time tools, runs 3 what-if scenarios
(order today / wait 7 / wait 14 days), computes decision_window_days.
Model: GPT-4o-mini.
"""

import json
import time
import logging
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agents.state import SupplyChainState
from audit.soc2_logger import log_agent_complete, log_tool_call, log_verification
from config import MODEL_MINI, OPENAI_API_KEY
from tools.supply_chain_tools import (
    get_demand_forecast,
    get_supplier_lead_time,
    run_scenario_model,
)

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are the Simulation Agent for Stellantis supply chain operations.
You receive scenario model outputs and must:
1. Recommend an order quantity that covers forecast + safety buffer
2. Perform self-verification that recommended qty >= forecast + safety_stock
3. Return JSON with keys:
   demand_forecast, lead_time_days, recommended_order_qty,
   simulation_scenarios (list), decision_window_days,
   verification_logs (list), decision_trail (list)

Verification format: "[SIMULATION] <check>: <value> → PASS ✓ / FAIL ✗"
If qty < forecast → FAIL ✗ → revise upward.
"""


def simulation_agent(state: SupplyChainState) -> SupplyChainState:
    t0 = time.time()
    run_id = state["run_id"]
    sku_id = state["sku_id"]
    supplier_id = state["supplier_id"]
    inventory_level = state.get("inventory_level", 0)
    safety_stock = state.get("safety_stock_threshold", 300)

    verification_logs = list(state.get("verification_logs", []))
    decision_trail = list(state.get("decision_trail", []))
    eval_metrics = dict(state.get("eval_metrics", {}))

    # ── Tool calls ────────────────────────────────────────────────────────────
    forecast_result = get_demand_forecast.invoke({"sku_id": sku_id, "days": 14})
    log_tool_call(run_id, "SIMULATION", "get_demand_forecast",
                  {"sku_id": sku_id, "days": 14}, forecast_result)

    lead_result = get_supplier_lead_time.invoke({"supplier_id": supplier_id})
    log_tool_call(run_id, "SIMULATION", "get_supplier_lead_time",
                  {"supplier_id": supplier_id}, lead_result)

    # Apply disruption penalty to lead time if active
    effective_lead_time = lead_result["lead_time_days"]
    if state.get("disruption_detected", False):
        disruption_sev = state.get("disruption_details", {}).get("severity", "none")
        if disruption_sev == "HIGH":
            effective_lead_time += 14
        elif disruption_sev == "MEDIUM":
            effective_lead_time += 7
        elif disruption_sev == "LOW":
            effective_lead_time += 2

    daily_demand = forecast_result.get("daily_demand", 30)
    forecast_14d = forecast_result.get("total_demand", daily_demand * 14)

    scenario_result = run_scenario_model.invoke({
        "sku_id": sku_id,
        "supplier_id": supplier_id,
        "current_inventory": inventory_level,
        "order_quantity": max(forecast_14d + safety_stock, 100),
        "daily_demand": daily_demand,
        "lead_time_days": effective_lead_time,
    })
    log_tool_call(run_id, "SIMULATION", "run_scenario_model",
                  {"sku_id": sku_id, "supplier_id": supplier_id}, scenario_result)

    # ── LLM reasoning ─────────────────────────────────────────────────────────
    llm = ChatOpenAI(model=MODEL_MINI, api_key=OPENAI_API_KEY, temperature=0)
    historical_trend = state.get("historical_trend", {})

    user_msg = f"""
Demand forecast (14 days): {json.dumps(forecast_result)}
Lead time data: {json.dumps(lead_result)}
Effective lead time (with disruption adjustment): {effective_lead_time} days
Current inventory: {inventory_level} units
Safety stock threshold: {safety_stock} units
Scenario model output: {json.dumps(scenario_result)}
Historical trend: {json.dumps(historical_trend)}

Determine recommended order quantity and perform self-verification.
"""
    messages = [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=user_msg)]
    response = llm.invoke(messages)
    raw = response.content.strip()
    tokens = response.usage_metadata.get("total_tokens", 0) if response.usage_metadata else 0

    try:
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        parsed: Dict[str, Any] = json.loads(raw)
    except Exception:
        # Fallback: compute directly
        recommended_qty = max(forecast_14d + safety_stock, inventory_level * 2)
        parsed = {
            "demand_forecast": forecast_14d,
            "lead_time_days": effective_lead_time,
            "recommended_order_qty": recommended_qty,
            "simulation_scenarios": scenario_result.get("scenarios", []),
            "decision_window_days": scenario_result.get("decision_window_days", 7),
            "verification_logs": [],
            "decision_trail": [],
        }

    recommended_qty = parsed.get("recommended_order_qty", forecast_14d + safety_stock)

    # ── Self-verification ─────────────────────────────────────────────────────
    covers_forecast = recommended_qty >= forecast_14d
    covers_with_buffer = recommended_qty >= forecast_14d + safety_stock * 0.5

    v1 = (
        f"[SIMULATION] Forecast 14 days: {forecast_14d} units"
    )
    v2 = (
        f"[VERIFY] Order qty {recommended_qty} covers forecast? "
        f"{recommended_qty} {'≥' if covers_forecast else '<'} {forecast_14d} → "
        f"{'PASS ✓' if covers_forecast else 'FAIL ✗ → revising upward'}"
    )
    v3 = (
        f"[VERIFY] Order qty {recommended_qty} covers forecast + safety buffer? "
        f"{'PASS ✓' if covers_with_buffer else 'FAIL ✗'}"
    )

    # Auto-correct if LLM under-recommended
    if not covers_forecast:
        recommended_qty = forecast_14d + safety_stock
        v2 += f" → revised to {recommended_qty}"

    log_verification(run_id, "SIMULATION", f"qty covers forecast ({recommended_qty} >= {forecast_14d})", covers_forecast)
    log_verification(run_id, "SIMULATION", f"qty covers forecast+buffer", covers_with_buffer)

    verification_logs += [v1, v2, v3] + parsed.get("verification_logs", [])
    decision_trail += parsed.get("decision_trail", [])
    decision_trail.append(
        f"Simulation: 14-day forecast={forecast_14d} units, lead_time={effective_lead_time}d, "
        f"recommended_order_qty={recommended_qty}, decision_window={scenario_result.get('decision_window_days', 7)} days."
    )

    cost_usd = (tokens / 1_000_000) * 0.60
    eval_metrics["simulation"] = {"latency_ms": int((time.time() - t0) * 1000), "tokens": tokens, "cost_usd": cost_usd}

    log_agent_complete(run_id, "SIMULATION",
                       {"demand_forecast": forecast_14d, "lead_time": effective_lead_time, "recommended_qty": recommended_qty},
                       cost_usd=cost_usd, tokens=tokens)

    return {
        **state,
        "demand_forecast": forecast_14d,
        "lead_time_days": effective_lead_time,
        "simulation_scenarios": scenario_result.get("scenarios", []),
        "decision_window_days": scenario_result.get("decision_window_days", 7),
        "verification_logs": verification_logs,
        "decision_trail": decision_trail,
        "eval_metrics": eval_metrics,
        "tokens_used": state.get("tokens_used", 0) + tokens,
        "cost_usd": state.get("cost_usd", 0.0) + cost_usd,
        # Store for optimization agent, cleared after consumption
        "_tool_cache": {
            **state.get("_tool_cache", {}),
            "recommended_order_qty": recommended_qty,
        },
    }
