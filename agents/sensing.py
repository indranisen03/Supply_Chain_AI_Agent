"""
Agent 1 — Sensing Agent

Calls get_inventory_level and check_supplier_disruption tools,
performs self-verification, and updates shared state.
Model: GPT-4o-mini (cost-efficient tool-calling).
"""

import json
import time
import logging
from typing import Any, Dict

from agents._llm import get_llm, token_cost
from langchain_core.messages import HumanMessage, SystemMessage

from agents.state import SupplyChainState
from audit.soc2_logger import log_agent_complete, log_tool_call, log_verification
from tools.supply_chain_tools import check_supplier_disruption, get_inventory_level

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are the Sensing Agent for Stellantis supply chain operations.
Your job is to retrieve current inventory levels and check for supplier disruptions.
After retrieving data, perform self-verification checks:
- Compare inventory_level against safety_stock_threshold
- Assess disruption severity

Return a JSON object with keys:
  inventory_level, safety_stock_threshold, unit_cost,
  disruption_detected, disruption_details,
  verification_logs (list of strings), decision_trail (list of strings)

Verification log format: "[SENSING] <check>: <value> → PASS ✓ / FAIL ✗"
"""


def sensing_agent(state: SupplyChainState) -> SupplyChainState:
    """Run Sensing Agent — mutates and returns updated state."""
    t0 = time.time()
    run_id = state["run_id"]
    sku_id = state["sku_id"]
    warehouse = state["warehouse"]
    supplier_id = state["supplier_id"]

    verification_logs = list(state.get("verification_logs", []))
    decision_trail = list(state.get("decision_trail", []))
    eval_metrics = dict(state.get("eval_metrics", {}))

    # ── Tool calls ────────────────────────────────────────────────────────────
    inv_result = get_inventory_level.invoke({"sku_id": sku_id, "warehouse": warehouse})
    log_tool_call(run_id, "SENSING", "get_inventory_level",
                  {"sku_id": sku_id, "warehouse": warehouse}, inv_result)

    dis_result = check_supplier_disruption.invoke({"supplier_id": supplier_id})
    log_tool_call(run_id, "SENSING", "check_supplier_disruption",
                  {"supplier_id": supplier_id}, dis_result)

    # ── LLM reasoning + self-verification ────────────────────────────────────
    llm = get_llm("mini")

    user_msg = f"""
Inventory data: {json.dumps(inv_result)}
Disruption data: {json.dumps(dis_result)}

Perform self-verification and return the required JSON.
"""
    messages = [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=user_msg)]

    response = llm.invoke(messages)
    raw = response.content.strip()
    tokens = response.usage_metadata.get("total_tokens", 0) if response.usage_metadata else 0

    # ── Parse LLM output ─────────────────────────────────────────────────────
    try:
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed: Dict[str, Any] = json.loads(raw.strip())
    except Exception:
        # Fallback: use raw tool results directly
        parsed = {
            "inventory_level": inv_result["inventory_level"],
            "safety_stock_threshold": inv_result["safety_stock_threshold"],
            "unit_cost": inv_result["unit_cost"],
            "disruption_detected": dis_result["disruption_active"],
            "disruption_details": dis_result,
            "verification_logs": [],
            "decision_trail": [],
        }

    # ── Deterministic self-verification (always run regardless of LLM) ───────
    inv_level = inv_result["inventory_level"]
    safety_stock = inv_result["safety_stock_threshold"]
    below_safety = inv_level < safety_stock

    v1 = f"[SENSING] Inventory {sku_id}: {inv_level} units"
    v2 = (
        f"[VERIFY] Safety stock check: {inv_level} < {safety_stock} → "
        f"{'FAIL ✗ → escalating to simulation' if below_safety else 'PASS ✓'}"
    )
    v3 = (
        f"[VERIFY] Supplier {supplier_id} disruption: {dis_result['severity']} → "
        f"{'FAIL ✗ → flagged' if dis_result['disruption_active'] else 'PASS ✓'}"
    )

    log_verification(run_id, "SENSING", f"inventory < safety_stock ({inv_level} < {safety_stock})", not below_safety)
    log_verification(run_id, "SENSING", f"supplier_disruption_active", not dis_result["disruption_active"])

    verification_logs += [v1, v2, v3]
    verification_logs += parsed.get("verification_logs", [])
    decision_trail += parsed.get("decision_trail", [])
    decision_trail.append(
        f"Sensing complete: {sku_id} has {inv_level} units "
        f"(safety stock {safety_stock}); "
        f"supplier {supplier_id} disruption={'YES' if dis_result['disruption_active'] else 'NO'}."
    )

    # ── Cost tracking ─────────────────────────────────────────────────────────
    cost_usd = token_cost(tokens, "mini")
    eval_metrics["sensing"] = {"latency_ms": int((time.time() - t0) * 1000), "tokens": tokens, "cost_usd": cost_usd}

    log_agent_complete(run_id, "SENSING",
                       {"inventory_level": inv_level, "disruption": dis_result["disruption_active"]},
                       cost_usd=cost_usd, tokens=tokens)

    return {
        **state,
        "inventory_level": inv_result["inventory_level"],
        "safety_stock_threshold": inv_result["safety_stock_threshold"],
        "unit_cost": inv_result["unit_cost"],
        "disruption_detected": dis_result["disruption_active"],
        "disruption_details": dis_result,
        "verification_logs": verification_logs,
        "decision_trail": decision_trail,
        "eval_metrics": eval_metrics,
        "tokens_used": state.get("tokens_used", 0) + tokens,
        "cost_usd": state.get("cost_usd", 0.0) + cost_usd,
    }
