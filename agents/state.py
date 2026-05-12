"""Shared LangGraph state — single source of truth across all 6 agents."""

from typing import TypedDict, Optional, List, Dict, Any
from pydantic import BaseModel, Field


class PurchaseOrderRecommendation(BaseModel):
    """Structured output from Optimization Agent."""
    part_id: str = Field(description="SKU/part identifier")
    quantity: int = Field(description="Recommended order quantity in units")
    required_by: str = Field(description="Required delivery date YYYY-MM-DD")
    supplier_id: str = Field(description="Preferred supplier ID")
    estimated_value: float = Field(description="Total PO value in USD")
    confidence: float = Field(ge=0.0, le=1.0, description="Model confidence 0-1")
    reasoning: str = Field(description="Plain English reasoning citing RAG sources")
    requires_human_approval: bool = Field(description="Whether HITL approval is required")
    hitl_tier: str = Field(description="AUTO | SOFT | HARD")
    decision_window_days: int = Field(description="Days until stockout risk > 50%")
    rag_sources: List[str] = Field(default_factory=list, description="RAG document citations")


class JudgeOutput(BaseModel):
    """Structured output from Validation Agent (Claude)."""
    quantity_justified: float = Field(ge=1.0, le=10.0)
    timeline_realistic: float = Field(ge=1.0, le=10.0)
    reasoning_grounded: float = Field(ge=1.0, le=10.0)
    overall_score: float = Field(ge=1.0, le=10.0)
    verdict: str = Field(description="PASS | FAIL")
    flags: List[str] = Field(default_factory=list)
    rationale: str = Field(description="Brief explanation of scores")


class SupplyChainState(TypedDict):
    # ── Core Identifiers ──────────────────────────────────────────────────────
    sku_id: str
    warehouse: str
    run_id: str          # SOC2 trace key; UUID set at pipeline entry
    supplier_id: str

    # ── Inventory ─────────────────────────────────────────────────────────────
    inventory_level: int
    safety_stock_threshold: int
    unit_cost: float

    # ── Disruption Detection ──────────────────────────────────────────────────
    disruption_detected: bool
    disruption_details: Dict[str, Any]

    # ── Historical Trend (Agent 2) ────────────────────────────────────────────
    historical_trend: Dict[str, Any]  # avg_demand_delta, supplier_on_time_rate

    # ── Forecast & Simulation (Agent 3) ──────────────────────────────────────
    demand_forecast: int
    lead_time_days: int
    simulation_scenarios: List[Dict[str, Any]]
    decision_window_days: int

    # ── Optimization Output (Agent 4) ─────────────────────────────────────────
    po_recommendation: Dict[str, Any]  # serialized PurchaseOrderRecommendation

    # ── HITL Gate ─────────────────────────────────────────────────────────────
    hitl_tier: str           # AUTO | SOFT | HARD
    human_approved: bool     # True ONLY when a human explicitly clicks Approve
    hitl_deadline: Optional[str]    # ISO-8601 deadline; AUTO=24hr, SOFT=12hr, HARD=None
    escalation_tier: Optional[str]  # Set when deadline exceeded, e.g. "AUTO→SOFT"
    hitl_notes: str          # Reviewer notes; mandatory for SOFT/HARD
    rejection_reason: Optional[str] # Populated when human_approved=False; logged to SOC2
    approver_role: Optional[str]    # Role/key used to approve; set by API, stored for SOC2

    # ── Pipeline Control ──────────────────────────────────────────────────────
    pipeline_halted: bool    # True = stop all downstream processing (cost cap, RAG failure)

    # ── Validation Judge (Agent 5) ────────────────────────────────────────────
    judge_score: float
    judge_verdict: str      # PASS | FAIL
    judge_flags: List[str]
    judge_scores_detail: Dict[str, float]

    # ── Summarizer (Agent 6) ──────────────────────────────────────────────────
    executive_summary: str

    # ── Audit & Verification ──────────────────────────────────────────────────
    verification_logs: List[str]   # "[AGENT] check: result ✓/✗"
    decision_trail: List[str]      # plain English reasoning per step

    # ── Observability ─────────────────────────────────────────────────────────
    eval_metrics: Dict[str, Any]   # {agent: {latency_ms, tokens, cost_usd}}
    tokens_used: int
    cost_usd: float

    # ── Error Handling ────────────────────────────────────────────────────────
    error: Optional[str]
    retry_count: int

    # ── Internal Cache (not persisted to HITL checkpoint) ────────────────────
    _tool_cache: Dict[str, Any]


def initial_state(
    sku_id: str,
    warehouse: str,
    run_id: str,
    supplier_id: str = "SUP-001",
    inventory_level: int = 0,
    demand_multiplier: float = 1.0,
) -> SupplyChainState:
    """Return a zeroed-out state for a new pipeline run."""
    return SupplyChainState(
        sku_id=sku_id,
        warehouse=warehouse,
        run_id=run_id,
        supplier_id=supplier_id,
        inventory_level=inventory_level,
        safety_stock_threshold=0,
        unit_cost=0.0,
        disruption_detected=False,
        disruption_details={},
        historical_trend={},
        demand_forecast=0,
        lead_time_days=0,
        simulation_scenarios=[],
        decision_window_days=0,
        po_recommendation={},
        hitl_tier="AUTO",
        human_approved=False,
        hitl_deadline=None,
        escalation_tier=None,
        hitl_notes="",
        rejection_reason=None,
        approver_role=None,
        pipeline_halted=False,
        judge_score=0.0,
        judge_verdict="",
        judge_flags=[],
        judge_scores_detail={},
        executive_summary="",
        verification_logs=[],
        decision_trail=[],
        eval_metrics={},
        tokens_used=0,
        cost_usd=0.0,
        error=None,
        retry_count=0,
        _tool_cache={},
    )
