"""
LangGraph orchestration for Stellantis Supply Chain AI.

Pipeline:
  sensing → historical_trend → simulation → optimization
  → hitl_gate (interrupt on HARD/SOFT) → validation → summarizer → done

MemorySaver is used exclusively for HITL checkpoint persistence.
Full state history is NOT persisted between runs (memory optimization).
"""

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from agents.historical_trend import historical_trend_agent
from agents.optimization import optimization_agent
from agents.sensing import sensing_agent
from agents.simulation import simulation_agent
from agents.state import SupplyChainState, initial_state
from agents.summarizer import summarizer_agent
from agents.validation import validation_agent
from audit.soc2_logger import log_agent_complete, log_event, log_human_decision
from config import MAX_AGENT_RETRIES, MAX_COST_PER_RUN_USD

logger = logging.getLogger(__name__)

# ── Self-healing wrapper ──────────────────────────────────────────────────────

def _debug_agent(error: str, state: SupplyChainState, agent_name: str) -> str:
    """Reason about what failed and return corrective guidance."""
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
    from config import MODEL_MINI, OPENAI_API_KEY
    try:
        llm = ChatOpenAI(model=MODEL_MINI, api_key=OPENAI_API_KEY, temperature=0, max_tokens=200)
        msg = f"Agent {agent_name} failed with error: {error}\nState keys: {list(state.keys())}\nWhat failed and what should be corrected? Answer in 1-2 sentences."
        response = llm.invoke([HumanMessage(content=msg)])
        return response.content.strip()
    except Exception:
        return f"Debug unavailable. Original error: {error}"


def _with_retry(agent_fn, agent_name: str):
    """Wrap an agent function with retry logic and SOC2 logging."""
    def wrapped(state: SupplyChainState) -> SupplyChainState:
        retry_count = state.get("retry_count", 0)
        for attempt in range(MAX_AGENT_RETRIES + 1):
            try:
                result = agent_fn(state)
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
                if attempt < MAX_AGENT_RETRIES:
                    guidance = _debug_agent(error_msg, state, agent_name)
                    vlog = list(state.get("verification_logs", []))
                    vlog.append(f"[RETRY] {agent_name} retrying (attempt {attempt + 2}) — {guidance}")
                    state = {**state, "verification_logs": vlog, "error": error_msg}
                    time.sleep(0.5 * (attempt + 1))
                else:
                    # Escalate to human
                    vlog = list(state.get("verification_logs", []))
                    vlog.append(f"[ERROR] {agent_name} failed after {MAX_AGENT_RETRIES + 1} attempts — escalating to human")
                    return {
                        **state,
                        "error": error_msg,
                        "verification_logs": vlog,
                        "retry_count": retry_count + MAX_AGENT_RETRIES + 1,
                    }
        return state
    return wrapped


# ── HITL Gate node ────────────────────────────────────────────────────────────

def hitl_gate_node(state: SupplyChainState) -> SupplyChainState:
    """
    Tiered Human-in-the-Loop gate.
    - AUTO: proceed immediately (human_approved already True)
    - SOFT: interrupt with countdown info; auto-approve after timeout
    - HARD: interrupt and block — requires explicit human approval
    """
    run_id = state["run_id"]
    tier = state.get("hitl_tier", "AUTO")
    po = state.get("po_recommendation", {})

    if tier == "AUTO":
        log_human_decision(run_id, tier, True, "Auto-approved (value < $10k)")
        vlog = list(state.get("verification_logs", []))
        vlog.append("[HITL] Tier AUTO — proceeding immediately ✓")
        return {**state, "human_approved": True, "verification_logs": vlog}

    # SOFT and HARD tiers trigger interrupt for human review
    interrupt_payload = {
        "tier": tier,
        "run_id": run_id,
        "po_recommendation": po,
        "hitl_deadline": state.get("hitl_deadline"),
        "message": (
            f"{'[SOFT] Proceed within 12 hours or this will auto-approve.' if tier == 'SOFT' else '[HARD] Explicit approval required — pipeline blocked.'}"
        ),
    }

    # This suspends graph execution until graph.invoke(Command(resume=...)) is called
    human_input: dict = interrupt(interrupt_payload)

    approved = human_input.get("approved", tier == "SOFT")  # SOFT defaults to approve if no input
    comment = human_input.get("comment", "")

    log_human_decision(run_id, tier, approved, comment)
    vlog = list(state.get("verification_logs", []))
    vlog.append(
        f"[HITL] Tier {tier} — human {'APPROVED ✓' if approved else 'REJECTED ✗'}"
        + (f" — {comment}" if comment else "")
    )
    trail = list(state.get("decision_trail", []))
    trail.append(f"HITL {tier}: human decision = {'approved' if approved else 'rejected'}.")

    return {
        **state,
        "human_approved": approved,
        "verification_logs": vlog,
        "decision_trail": trail,
    }


# ── Cost guardrail node ───────────────────────────────────────────────────────

def cost_check_node(state: SupplyChainState) -> SupplyChainState:
    """Emit warning if cumulative cost exceeds cap."""
    cost = state.get("cost_usd", 0.0)
    vlog = list(state.get("verification_logs", []))
    if cost > MAX_COST_PER_RUN_USD:
        msg = f"[COST] WARNING: run cost ${cost:.4f} exceeds cap ${MAX_COST_PER_RUN_USD:.2f}"
        logger.warning(msg)
        vlog.append(msg)
        log_event(state["run_id"], "COST_GUARD", "cost_exceeded",
                  {"cost_usd": cost}, {"cap": MAX_COST_PER_RUN_USD}, verification_passed=False)
    return {**state, "verification_logs": vlog}


# ── SOC2 finalization node ────────────────────────────────────────────────────

def soc2_finalize_node(state: SupplyChainState) -> SupplyChainState:
    """Write final SOC2 audit record for completed run."""
    log_event(
        run_id=state["run_id"],
        agent="PIPELINE",
        action="run_complete",
        input_summary={"sku_id": state["sku_id"], "warehouse": state["warehouse"]},
        output_summary={
            "hitl_tier": state.get("hitl_tier"),
            "human_approved": state.get("human_approved"),
            "judge_verdict": state.get("judge_verdict"),
            "judge_score": state.get("judge_score"),
            "cost_usd": state.get("cost_usd"),
            "tokens_used": state.get("tokens_used"),
        },
        verification_passed=state.get("judge_verdict") == "PASS",
        cost_usd=state.get("cost_usd", 0.0),
        tokens_used=state.get("tokens_used", 0),
    )
    return state


# ── Routing logic ─────────────────────────────────────────────────────────────

def should_continue_after_hitl(state: SupplyChainState) -> Literal["validation", "end"]:
    """After HITL gate: proceed to validation only if approved."""
    if state.get("human_approved", False):
        return "validation"
    return "end"


def should_handle_error(state: SupplyChainState) -> Literal["sensing", "historical_trend", "simulation", "optimization", "hitl_gate", "validation", "summarizer", "end"]:
    """If an unrecoverable error occurred, skip to end."""
    if state.get("error") and state.get("retry_count", 0) > MAX_AGENT_RETRIES:
        return "end"
    return "sensing"


# ── Build the graph ───────────────────────────────────────────────────────────

def build_graph() -> tuple:
    """
    Returns (compiled_graph, checkpointer).
    The checkpointer enables HITL interruption and resumption.
    """
    checkpointer = MemorySaver()
    workflow = StateGraph(SupplyChainState)

    # Register nodes (all wrapped with retry + self-healing)
    workflow.add_node("sensing", _with_retry(sensing_agent, "SENSING"))
    workflow.add_node("historical_trend", _with_retry(historical_trend_agent, "HISTORICAL_TREND"))
    workflow.add_node("simulation", _with_retry(simulation_agent, "SIMULATION"))
    workflow.add_node("optimization", _with_retry(optimization_agent, "OPTIMIZATION"))
    workflow.add_node("hitl_gate", hitl_gate_node)
    workflow.add_node("cost_check", cost_check_node)
    workflow.add_node("validation", _with_retry(validation_agent, "VALIDATION"))
    workflow.add_node("summarizer", _with_retry(summarizer_agent, "SUMMARIZER"))
    workflow.add_node("soc2_finalize", soc2_finalize_node)

    # Define pipeline flow
    workflow.set_entry_point("sensing")
    workflow.add_edge("sensing", "historical_trend")
    workflow.add_edge("historical_trend", "simulation")
    workflow.add_edge("simulation", "optimization")
    workflow.add_edge("optimization", "cost_check")
    workflow.add_edge("cost_check", "hitl_gate")

    # Conditional routing after HITL
    workflow.add_conditional_edges(
        "hitl_gate",
        should_continue_after_hitl,
        {"validation": "validation", "end": "soc2_finalize"},
    )

    workflow.add_edge("validation", "summarizer")
    workflow.add_edge("summarizer", "soc2_finalize")
    workflow.add_edge("soc2_finalize", END)

    graph = workflow.compile(checkpointer=checkpointer, interrupt_before=["hitl_gate"])
    return graph, checkpointer


# ── Public API ────────────────────────────────────────────────────────────────

def run_pipeline(
    sku_id: str = "SKU-4821",
    warehouse: str = "WH-Detroit",
    supplier_id: str = "SUP-001",
    thread_id: str | None = None,
    inventory_override: int | None = None,
) -> tuple[SupplyChainState, str]:
    """
    Run the full pipeline synchronously.
    Returns (final_state, run_id).
    Handles HITL AUTO tier automatically.
    For SOFT/HARD, returns intermediate state and run_id for resumption.
    """
    from tools.supply_chain_tools import clear_cache
    clear_cache()

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

    graph, _ = build_graph()
    config = {"configurable": {"thread_id": thread_id}}

    # Run until HITL gate
    events = list(graph.stream(state, config))
    last_state = events[-1] if events else state

    # Extract state from graph checkpoint
    checkpoint_state = graph.get_state(config)
    current_state = checkpoint_state.values if checkpoint_state else last_state

    # If HITL tier is AUTO, the interrupt was already handled — resume immediately
    tier = current_state.get("hitl_tier", "AUTO")
    if tier == "AUTO":
        from langgraph.types import Command
        events = list(graph.stream(Command(resume={"approved": True, "comment": "AUTO"}), config))
        checkpoint_state = graph.get_state(config)
        current_state = checkpoint_state.values if checkpoint_state else current_state

    return current_state, run_id


def resume_pipeline(thread_id: str, approved: bool, comment: str = "") -> SupplyChainState:
    """Resume a SOFT or HARD HITL-paused pipeline with human decision."""
    from langgraph.types import Command
    graph, _ = build_graph()
    config = {"configurable": {"thread_id": thread_id}}
    events = list(graph.stream(Command(resume={"approved": approved, "comment": comment}), config))
    checkpoint_state = graph.get_state(config)
    return checkpoint_state.values if checkpoint_state else {}
