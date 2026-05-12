"""
LangGraph orchestration for Supply Chain AI.

Pipeline:
  sensing → historical_trend → simulation → optimization
  → cost_check → validation → hitl_gate (interrupt on all tiers) → summarizer → done

MemorySaver is used exclusively for HITL checkpoint persistence.
Full state history is NOT persisted between runs (memory optimization).
"""

import json
import logging
import time
import uuid
import numpy as np
from datetime import datetime, timezone
from typing import Any, Dict, Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from agents.historical_trend import historical_trend_agent
from agents.optimization import optimization_agent
from agents.policy import AGENT_MAX_RETRIES
from agents.sensing import sensing_agent
from agents.simulation import simulation_agent
from agents.state import SupplyChainState, initial_state
from agents.summarizer import summarizer_agent
from agents.validation import validation_agent
from audit.soc2_logger import log_agent_complete, log_event, log_human_decision
from config import MAX_AGENT_RETRIES as DEFAULT_MAX_RETRIES, MAX_COST_PER_RUN_USD

logger = logging.getLogger(__name__)

# ── Self-healing wrapper ──────────────────────────────────────────────────────

def _debug_agent(error: str, state: SupplyChainState, agent_name: str) -> str:
    """Reason about what failed and return corrective guidance."""
    from langchain_core.messages import HumanMessage
    from agents._llm import get_llm
    try:
        llm = get_llm("mini", max_tokens=200)
        msg = f"Agent {agent_name} failed with error: {error}\nState keys: {list(state.keys())}\nWhat failed and what should be corrected? Answer in 1-2 sentences."
        response = llm.invoke([HumanMessage(content=msg)])
        return response.content.strip()
    except Exception:
        return f"Debug unavailable. Original error: {error}"


def _with_retry(agent_fn, agent_name: str):
    """Wrap an agent function with per-agent retry logic and SOC2 logging."""
    max_retries = AGENT_MAX_RETRIES.get(agent_name, DEFAULT_MAX_RETRIES)

    def wrapped(state: SupplyChainState) -> SupplyChainState:
        # Skip immediately if pipeline already halted
        if state.get("pipeline_halted"):
            return state

        retry_count = state.get("retry_count", 0)
        for attempt in range(max_retries + 1):
            try:
                result = _sanitize(agent_fn(state))
                if attempt > 0:
                    vlog = list(result.get("verification_logs", []))
                    vlog.append(f"[RETRY] {agent_name} succeeded on attempt {attempt + 1}")
                    result = {**result, "verification_logs": vlog, "retry_count": 0}
                return result
            except Exception as exc:
                error_msg = str(exc)
                logger.error("[%s] attempt %d failed: %s", agent_name, attempt + 1, error_msg)
                log_event(
                    state.get("run_id", "unknown"),
                    agent_name,
                    f"retry_attempt_{attempt + 1}",
                    {"error": error_msg},
                    {"attempt": attempt + 1},
                    verification_passed=False,
                )
                if attempt < max_retries:
                    guidance = _debug_agent(error_msg, state, agent_name)
                    vlog = list(state.get("verification_logs", []))
                    vlog.append(f"[RETRY] {agent_name} retrying (attempt {attempt + 2}) — {guidance}")
                    state = {**state, "verification_logs": vlog, "error": error_msg}
                    time.sleep(0.5 * (attempt + 1))
                else:
                    vlog = list(state.get("verification_logs", []))
                    vlog.append(f"[ERROR] {agent_name} failed after {max_retries + 1} attempts — escalating to HARD")
                    # VALIDATION failures escalate to HARD; others halt pipeline
                    if agent_name == "VALIDATION":
                        return {
                            **state,
                            "error": error_msg,
                            "hitl_tier": "HARD",
                            "verification_logs": vlog,
                            "retry_count": retry_count + max_retries + 1,
                        }
                    return {
                        **state,
                        "error": error_msg,
                        "pipeline_halted": True,
                        "verification_logs": vlog,
                        "retry_count": retry_count + max_retries + 1,
                    }
        return state
    return wrapped


# ── HITL Gate node ────────────────────────────────────────────────────────────

def hitl_gate_node(state: SupplyChainState) -> SupplyChainState:
    """
    Tiered Human-in-the-Loop gate. ALL tiers interrupt — agents NEVER
    autonomously execute purchases.

    - AUTO  (<$10k):  fast-track, 24hr escalation window to SOFT
    - SOFT  ($10-50k): 12hr review window, escalates to HARD on timeout
    - HARD  (>$50k):  hard block, no timeout, explicit approval only

    human_approved is ONLY set True when a human explicitly clicks Approve.
    """
    run_id = state["run_id"]
    tier = state.get("hitl_tier", "AUTO")
    po = state.get("po_recommendation", {})

    _tier_messages = {
        "AUTO": "[AUTO] Fast-track approval required. No action in 24hrs will escalate to SOFT.",
        "SOFT": "[SOFT] Review required within 12hrs. No action will escalate to HARD.",
        "HARD": "[HARD] Pipeline blocked. Explicit approval required — no timeout fallback.",
    }

    interrupt_payload = {
        "tier": tier,
        "run_id": run_id,
        "po_recommendation": po,
        "hitl_deadline": state.get("hitl_deadline"),
        "judge_score": state.get("judge_score"),
        "judge_verdict": state.get("judge_verdict"),
        "message": _tier_messages.get(tier, ""),
    }

    # Suspends graph execution until resume_pipeline() is called
    human_input: dict = interrupt(interrupt_payload)

    approved = human_input.get("approved", False)
    notes = human_input.get("notes", "")
    comment = human_input.get("comment", notes)
    rejection_reason = human_input.get("rejection_reason", "") if not approved else None
    approver_role = human_input.get("approver_role", None)

    # Check if deadline was exceeded — record escalation, never auto-approve
    escalation_tier = state.get("escalation_tier")
    deadline_str = state.get("hitl_deadline")
    if deadline_str:
        try:
            deadline = datetime.fromisoformat(deadline_str)
            if datetime.now() > deadline:
                if tier == "AUTO":
                    escalation_tier = "AUTO→SOFT"
                elif tier == "SOFT":
                    escalation_tier = "SOFT→HARD"
        except ValueError:
            pass

    log_human_decision(
        run_id, tier, approved,
        notes=notes,
        escalation_tier=escalation_tier,
        rejection_reason=rejection_reason,
    )

    vlog = list(state.get("verification_logs", []))
    esc_tag = f" [escalated from {escalation_tier}]" if escalation_tier else ""
    vlog.append(
        f"[HITL] Tier {tier}{esc_tag} — human {'APPROVED ✓' if approved else 'REJECTED ✗'}"
        + (f" — {notes}" if notes else "")
    )
    if rejection_reason:
        vlog.append(f"[HITL] Rejection reason: {rejection_reason}")

    trail = list(state.get("decision_trail", []))
    trail.append(f"HITL {tier}{esc_tag}: human decision = {'approved' if approved else 'rejected'}.")

    return {
        **state,
        "human_approved": approved,
        "hitl_notes": notes,
        "rejection_reason": rejection_reason,
        "approver_role": approver_role,
        "escalation_tier": escalation_tier,
        "verification_logs": vlog,
        "decision_trail": trail,
    }


# ── Cost guardrail node ───────────────────────────────────────────────────────

def cost_check_node(state: SupplyChainState) -> SupplyChainState:
    """Hard stop if cumulative cost exceeds cap — sets pipeline_halted=True."""
    cost = state.get("cost_usd", 0.0)
    vlog = list(state.get("verification_logs", []))
    if cost > MAX_COST_PER_RUN_USD:
        msg = f"[COST] HARD STOP: run cost ${cost:.4f} exceeds cap ${MAX_COST_PER_RUN_USD:.2f} — pipeline halted"
        logger.error(msg)
        vlog.append(msg)
        log_event(
            state["run_id"], "COST_GUARD", "cost_exceeded",
            {"cost_usd": cost}, {"cap": MAX_COST_PER_RUN_USD},
            verification_passed=False,
        )
        return {**state, "pipeline_halted": True, "error": msg, "verification_logs": vlog}
    return {**state, "verification_logs": vlog}


# ── SOC2 finalization node ────────────────────────────────────────────────────

def soc2_finalize_node(state: SupplyChainState) -> SupplyChainState:
    """Write final SOC2 audit record and update procedural memory."""
    log_event(
        run_id=state["run_id"],
        agent="PIPELINE",
        action="run_complete",
        input_summary={"sku_id": state["sku_id"], "warehouse": state["warehouse"]},
        output_summary={
            "hitl_tier": state.get("hitl_tier"),
            "human_approved": state.get("human_approved"),
            "rejection_reason": state.get("rejection_reason"),
            "judge_verdict": state.get("judge_verdict"),
            "judge_score": state.get("judge_score"),
            "cost_usd": state.get("cost_usd"),
            "tokens_used": state.get("tokens_used"),
            "pipeline_halted": state.get("pipeline_halted"),
        },
        verification_passed=state.get("judge_verdict") == "PASS",
        cost_usd=state.get("cost_usd", 0.0),
        tokens_used=state.get("tokens_used", 0),
    )

    # Update procedural memories if pipeline ran to completion
    if not state.get("pipeline_halted") and state.get("po_recommendation"):
        po = state["po_recommendation"]
        try:
            from memory.supplier_memory import update_supplier_memory
            hist = state.get("historical_trend", {})
            update_supplier_memory(
                supplier_id=state.get("supplier_id", "unknown"),
                run_id=state["run_id"],
                on_time=hist.get("supplier_on_time_rate", 0.9) >= 0.85,
                lead_time_days=state.get("lead_time_days", 14),
                disruption_detected=state.get("disruption_detected", False),
                po_value=po.get("estimated_value", 0.0),
            )
        except Exception as exc:
            logger.warning("supplier_memory update failed: %s", exc)

        try:
            from memory.hitl_memory import record_hitl_decision
            record_hitl_decision(
                run_id=state["run_id"],
                tier=state.get("hitl_tier", "AUTO"),
                approved=state.get("human_approved", False),
                po_value=po.get("estimated_value", 0.0),
                judge_score=state.get("judge_score", 0.0),
                rejection_reason=state.get("rejection_reason"),
            )
        except Exception as exc:
            logger.warning("hitl_memory update failed: %s", exc)

    return state


# ── Routing logic ─────────────────────────────────────────────────────────────

def route_after_cost_check(state: SupplyChainState) -> Literal["validation", "soc2_finalize"]:
    """After cost check: validation if healthy, soc2_finalize if halted."""
    if state.get("pipeline_halted"):
        return "soc2_finalize"
    return "validation"


def route_after_validation(state: SupplyChainState) -> Literal["hitl_gate", "soc2_finalize"]:
    """After validation: hitl_gate always (judge may have escalated tier)."""
    if state.get("pipeline_halted"):
        return "soc2_finalize"
    return "hitl_gate"


def should_continue_after_hitl(state: SupplyChainState) -> Literal["summarizer", "soc2_finalize"]:
    """After HITL gate: summarizer only if approved."""
    if state.get("human_approved", False):
        return "summarizer"
    return "soc2_finalize"


# ── Build the graph ───────────────────────────────────────────────────────────

def _sanitize(obj):
    """Recursively convert numpy scalar types to Python native so MemorySaver can serialize."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def build_graph() -> tuple:
    """
    Returns (compiled_graph, checkpointer).
    The checkpointer enables HITL interruption and resumption.

    Pipeline order: sensing → hist_trend → simulation → optimization
                    → cost_check → validation → hitl_gate → summarizer → soc2_finalize
    """
    checkpointer = MemorySaver()
    workflow = StateGraph(SupplyChainState)

    # Register nodes (all wrapped with retry + self-healing, except gates)
    workflow.add_node("sensing", _with_retry(sensing_agent, "SENSING"))
    workflow.add_node("hist_trend", _with_retry(historical_trend_agent, "HISTORICAL_TREND"))
    workflow.add_node("simulation", _with_retry(simulation_agent, "SIMULATION"))
    workflow.add_node("optimization", _with_retry(optimization_agent, "OPTIMIZATION"))
    workflow.add_node("cost_check", cost_check_node)
    workflow.add_node("validation", _with_retry(validation_agent, "VALIDATION"))
    workflow.add_node("hitl_gate", hitl_gate_node)
    workflow.add_node("summarizer", _with_retry(summarizer_agent, "SUMMARIZER"))
    workflow.add_node("soc2_finalize", soc2_finalize_node)

    # Define pipeline flow
    workflow.set_entry_point("sensing")
    workflow.add_edge("sensing", "hist_trend")
    workflow.add_edge("hist_trend", "simulation")
    workflow.add_edge("simulation", "optimization")
    workflow.add_edge("optimization", "cost_check")

    # Conditional routing after cost check (halt → soc2_finalize)
    workflow.add_conditional_edges(
        "cost_check",
        route_after_cost_check,
        {"validation": "validation", "soc2_finalize": "soc2_finalize"},
    )

    # Conditional routing after validation (halt → soc2_finalize, otherwise → hitl_gate)
    workflow.add_conditional_edges(
        "validation",
        route_after_validation,
        {"hitl_gate": "hitl_gate", "soc2_finalize": "soc2_finalize"},
    )

    # Conditional routing after HITL (approved → summarizer, rejected → soc2_finalize)
    workflow.add_conditional_edges(
        "hitl_gate",
        should_continue_after_hitl,
        {"summarizer": "summarizer", "soc2_finalize": "soc2_finalize"},
    )

    workflow.add_edge("summarizer", "soc2_finalize")
    workflow.add_edge("soc2_finalize", END)

    graph = workflow.compile(checkpointer=checkpointer, interrupt_before=["hitl_gate"])
    return graph, checkpointer


# ── Singleton graph (shared checkpointer so resume works) ────────────────────
_CHECKPOINTER = MemorySaver()
_GRAPH = None


def get_graph():
    """Return the module-level singleton compiled graph."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH, _ = build_graph()
        # Swap in the module-level checkpointer so run + resume share state
        _GRAPH.checkpointer = _CHECKPOINTER
    return _GRAPH


# ── Public API ────────────────────────────────────────────────────────────────

def run_pipeline(
    sku_id: str = "SKU-0000",
    warehouse: str = "WH-Detroit",
    supplier_id: str = "SUP-003",
    thread_id: str | None = None,
    inventory_override: int | None = None,
) -> tuple[SupplyChainState, str]:
    """
    Run the full pipeline synchronously up to the HITL gate.
    Always pauses for human confirmation — no tier is auto-approved.
    Returns (paused_state, run_id).
    """
    from tools.supply_chain_tools import clear_cache
    from rag.retriever import clear_query_cache
    clear_cache()
    clear_query_cache()

    run_id = str(uuid.uuid4())
    thread_id = thread_id or run_id

    state = initial_state(
        sku_id=sku_id,
        warehouse=warehouse,
        run_id=run_id,
        supplier_id=supplier_id,
    )
    if inventory_override is not None:
        state = {**state, "inventory_level": inventory_override}

    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    # Run until interrupt_before=["hitl_gate"] fires — always pauses here
    for _ in graph.stream(state, config):
        pass

    checkpoint_state = graph.get_state(config)
    current_state = dict(checkpoint_state.values) if checkpoint_state else dict(state)

    return current_state, run_id


def resume_pipeline(
    thread_id: str,
    approved: bool,
    notes: str = "",
    comment: str = "",
    rejection_reason: str = "",
    approver_role: str = "",
) -> SupplyChainState:
    """Resume any HITL-paused pipeline with an explicit human decision."""
    from langgraph.types import Command
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    resume_payload = {
        "approved": approved,
        "notes": notes,
        "comment": comment or notes,
        "rejection_reason": rejection_reason if not approved else "",
        "approver_role": approver_role,
    }
    list(graph.stream(Command(resume=resume_payload), config))
    checkpoint_state = graph.get_state(config)
    return checkpoint_state.values if checkpoint_state else {}
