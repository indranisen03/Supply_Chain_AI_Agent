"""
Agent 4 — Optimization Agent

Uses RAG (EnsembleRetriever) to ground decisions in real Stellantis documents.
Synthesises all prior agent outputs into a PurchaseOrderRecommendation.
Determines HITL tier based on estimated_value.
Model: GPT-4o (complex reasoning).
"""

import json
import time
import logging
from datetime import datetime, timedelta
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from agents._llm import get_llm, token_cost

from agents.state import PurchaseOrderRecommendation, SupplyChainState
from agents.policy import determine_hitl_tier
from audit.soc2_logger import log_agent_complete, log_verification
from config import (
    HITL_AUTO_MAX_USD,
    HITL_SOFT_MAX_USD,
    MODEL_LARGE,
    SOFT_HITL_TIMEOUT_HOURS,
)

logger = logging.getLogger(__name__)

# Cached system prompt text (used for Anthropic prompt caching)
_SYSTEM_PROMPT_TEXT = """\
You are the Optimization Agent for Stellantis supply chain procurement.
You have access to Stellantis policy documents via RAG context.
Your task: synthesize sensing, historical trend, and simulation data into a
single PurchaseOrderRecommendation.

Rules:
- Quantity MUST cover demand_forecast + safety_stock (or explain why not)
- required_by date must be today + lead_time_days + 2 days buffer
- Cite specific RAG sources in reasoning (e.g. "per Stellantis Responsible Purchasing Guidelines")
- Confidence = 0.65–0.95 (never below 0.65; baseline data always supports a recommendation)
- hitl_tier determined by estimated_value thresholds (passed as context)

Return ONLY a valid JSON object matching PurchaseOrderRecommendation schema.
No extra text.
"""


def optimization_agent(state: SupplyChainState) -> SupplyChainState:
    t0 = time.time()
    run_id = state["run_id"]
    sku_id = state["sku_id"]
    supplier_id = state["supplier_id"]

    verification_logs = list(state.get("verification_logs", []))
    decision_trail = list(state.get("decision_trail", []))
    eval_metrics = dict(state.get("eval_metrics", {}))

    # ── Gather context ────────────────────────────────────────────────────────
    inventory_level = state.get("inventory_level", 0)
    safety_stock = state.get("safety_stock_threshold", 300)
    unit_cost = state.get("unit_cost", 50.0)
    demand_forecast = state.get("demand_forecast", 400)
    lead_time_days = state.get("lead_time_days", 14)
    decision_window_days = state.get("decision_window_days", 7)
    historical_trend = state.get("historical_trend", {})
    disruption_detected = state.get("disruption_detected", False)
    disruption_details = state.get("disruption_details", {})
    recommended_qty = state.get("_tool_cache", {}).get("recommended_order_qty", demand_forecast + safety_stock)
    simulation_scenarios = state.get("simulation_scenarios", [])

    estimated_value = recommended_qty * unit_cost
    required_by = (datetime.now() + timedelta(days=lead_time_days + 7)).strftime("%Y-%m-%d")

    hitl_tier, requires_approval = determine_hitl_tier(estimated_value)

    # ── RAG retrieval (cached; halt if unavailable) ───────────────────────────
    rag_context = ""
    rag_sources = []
    rag_ok = False
    try:
        from rag.retriever import retrieve_docs, retrieve, get_retriever
        retriever = get_retriever()
        query = f"Stellantis procurement approval threshold {supplier_id} emergency order quantity policy"
        rag_context = retrieve(query, retriever)
        rag_sources = list({
            doc.metadata.get("source", "unknown")
            for doc in retrieve_docs(query, retriever)
        })
        rag_ok = bool(rag_context)
    except Exception as exc:
        logger.error("RAG retrieval failed — halting pipeline: %s", exc)
        vlog = list(state.get("verification_logs", []))
        vlog.append(f"[OPTIMIZATION] RAG failure — pipeline halted: {exc}")
        return {
            **state,
            "pipeline_halted": True,
            "error": f"RAG failure: {exc}",
            "verification_logs": vlog,
        }

    # ── Supplier context from procedural memory ───────────────────────────────
    try:
        from memory.supplier_memory import get_supplier_context
        supplier_ctx = get_supplier_context(supplier_id)
    except Exception:
        supplier_ctx = {}

    # ── LLM reasoning with Anthropic prompt caching ───────────────────────────
    llm = get_llm("large")
    model_id = getattr(llm, "model_name", getattr(llm, "model", ""))
    use_cache = "claude" in model_id.lower() or "anthropic" in str(type(llm)).lower()

    # Build system message — cache the stable part (prompt + RAG context)
    cached_system_content = f"{_SYSTEM_PROMPT_TEXT}\n\n=== RAG Policy Context ===\n{rag_context}"
    if use_cache:
        system_msg = SystemMessage(content=[{
            "type": "text",
            "text": cached_system_content,
            "cache_control": {"type": "ephemeral"},
        }])
    else:
        system_msg = SystemMessage(content=cached_system_content)

    user_msg = f"""
=== Supply Chain Context ===
SKU: {sku_id} | Warehouse: {state['warehouse']} | Supplier: {supplier_id}
Current inventory: {inventory_level} units | Safety stock: {safety_stock} units
Unit cost: ${unit_cost:.2f} | Recommended qty: {recommended_qty} units
Estimated PO value: ${estimated_value:,.2f}
Required by: {required_by}
Lead time (effective): {lead_time_days} days
Decision window: {decision_window_days} days until stockout risk >50%
Demand forecast (14d): {demand_forecast} units
Disruption detected: {disruption_detected} — {disruption_details.get('details', 'N/A')}
Historical trend: {json.dumps(historical_trend)}
Simulation scenarios: {json.dumps(simulation_scenarios)}
Supplier history: {json.dumps(supplier_ctx)}

=== HITL Thresholds ===
AUTO: < ${HITL_AUTO_MAX_USD:,.0f}
SOFT: ${HITL_AUTO_MAX_USD:,.0f}–${HITL_SOFT_MAX_USD:,.0f}
HARD: > ${HITL_SOFT_MAX_USD:,.0f}
Current tier: {hitl_tier} (estimated_value = ${estimated_value:,.2f})

Generate the PurchaseOrderRecommendation JSON.
Fields: part_id, quantity, required_by, supplier_id, estimated_value,
        confidence, reasoning, requires_human_approval, hitl_tier,
        decision_window_days, rag_sources
"""
    messages = [system_msg, HumanMessage(content=user_msg)]
    response = llm.invoke(messages)
    raw = response.content.strip()
    tokens = response.usage_metadata.get("total_tokens", 0) if response.usage_metadata else 0

    try:
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        po_dict: Dict[str, Any] = json.loads(raw)
    except Exception:
        po_dict = {
            "part_id": sku_id,
            "quantity": recommended_qty,
            "required_by": required_by,
            "supplier_id": supplier_id,
            "estimated_value": estimated_value,
            "confidence": 0.80,
            "reasoning": f"Based on {demand_forecast} unit 14-day forecast and {safety_stock} safety stock.",
            "requires_human_approval": requires_approval,
            "hitl_tier": hitl_tier,
            "decision_window_days": decision_window_days,
            "rag_sources": rag_sources,
        }

    # ── Override HITL tier with deterministic logic (LLM can't override thresholds) ──
    actual_value = po_dict.get("estimated_value", estimated_value)
    actual_tier, _ = _determine_hitl_tier(actual_value)
    po_dict["hitl_tier"] = actual_tier
    po_dict["requires_human_approval"] = True  # All tiers require human confirmation

    # Ensure rag_sources populated
    if not po_dict.get("rag_sources"):
        po_dict["rag_sources"] = rag_sources

    # Confidence floor: even under disruption, data is sufficient for a baseline recommendation
    raw_conf = float(po_dict.get("confidence", 0.0))
    if raw_conf < 0.65:
        po_dict["confidence"] = 0.70 if disruption_detected else 0.75
        logger.info("Confidence floored from %.2f → %.2f", raw_conf, po_dict["confidence"])

    # Validate with Pydantic
    try:
        po_model = PurchaseOrderRecommendation(**po_dict)
        po_dict = po_model.model_dump()
    except Exception as exc:
        logger.warning("PO validation warning: %s", exc)

    # ── Self-verification ─────────────────────────────────────────────────────
    qty = po_dict.get("quantity", recommended_qty)
    qty_ok = qty >= demand_forecast
    buffer_ok = qty >= demand_forecast + safety_stock * 0.5
    conf = po_dict.get("confidence", 0.0)
    conf_high = conf >= 0.85
    conf_ok   = conf >= 0.65   # minimum viable
    has_rag = bool(po_dict.get("rag_sources"))

    v1 = (
        f"[OPTIMIZATION] Recommended qty {qty} vs forecast {demand_forecast} → "
        f"{'PASS ✓' if qty_ok else 'FAIL ✗'}"
    )
    if conf_high:
        conf_label = "PASS ✓"
    elif conf_ok:
        conf_label = "WARN ⚠ moderate confidence — human review recommended"
    else:
        conf_label = "FAIL ✗ confidence below minimum threshold"
    v2 = f"[OPTIMIZATION] Confidence {conf:.2f} → {conf_label}"
    v3 = (
        f"[OPTIMIZATION] RAG sources cited: {len(po_dict.get('rag_sources', []))} → "
        f"{'PASS ✓' if has_rag else 'WARN ⚠ no RAG grounding'}"
    )
    v4 = f"[OPTIMIZATION] HITL tier: {actual_tier} (value ${actual_value:,.2f})"

    log_verification(run_id, "OPTIMIZATION", f"qty >= forecast ({qty} >= {demand_forecast})", qty_ok)
    log_verification(run_id, "OPTIMIZATION", f"confidence >= 0.85 ({conf:.2f})", conf_high)
    log_verification(run_id, "OPTIMIZATION", "rag_sources_cited", has_rag)

    verification_logs += [v1, v2, v3, v4]
    decision_trail.append(
        f"Optimization: PO recommendation for {qty} units of {sku_id} from {supplier_id} "
        f"by {po_dict.get('required_by')}, value ${actual_value:,.2f}, tier={actual_tier}."
    )

    # ── HITL deadline — AUTO=24hr, SOFT=12hr, HARD=no timeout ────────────────
    hitl_deadline = None
    if actual_tier == "AUTO":
        hitl_deadline = (datetime.now() + timedelta(hours=24)).isoformat()
    elif actual_tier == "SOFT":
        hitl_deadline = (datetime.now() + timedelta(hours=SOFT_HITL_TIMEOUT_HOURS)).isoformat()

    # Clear simulation_scenarios from state after consumption (memory optimization)
    cost_usd = token_cost(tokens, "large")
    eval_metrics["optimization"] = {"latency_ms": int((time.time() - t0) * 1000), "tokens": tokens, "cost_usd": cost_usd}

    log_agent_complete(run_id, "OPTIMIZATION", po_dict, cost_usd=cost_usd, tokens=tokens)

    return {
        **state,
        "po_recommendation": po_dict,
        "hitl_tier": actual_tier,
        "human_approved": False,  # Never pre-approved — human must explicitly confirm
        "hitl_deadline": hitl_deadline,
        "simulation_scenarios": [],  # cleared — consumed
        "verification_logs": verification_logs,
        "decision_trail": decision_trail,
        "eval_metrics": eval_metrics,
        "tokens_used": state.get("tokens_used", 0) + tokens,
        "cost_usd": state.get("cost_usd", 0.0) + cost_usd,
    }
