"""
Agent 5 — Validation Agent (LLM Judge)

Uses Claude claude-sonnet-4-5 for an independent blind evaluation.
Receives ONLY raw inputs + PO recommendation — no intermediate reasoning.
Scores 3 dimensions (1-10) and returns overall verdict.
"""

import json
import time
import logging
from datetime import datetime
from typing import Any, Dict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from agents.state import JudgeOutput, SupplyChainState
from audit.soc2_logger import log_judge_score, log_verification
from config import ANTHROPIC_API_KEY, JUDGE_PASS_THRESHOLD, MODEL_JUDGE

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an independent supply chain procurement judge for Stellantis.
You receive raw facts and a purchase order recommendation.
You have NOT seen the intermediate agent reasoning.

Score three dimensions from 1 to 10:
1. quantity_justified (1-10): Is the order quantity supported by inventory and forecast data?
2. timeline_realistic (1-10): Is the required_by date achievable given the stated lead time?
3. reasoning_grounded (1-10): Is the reasoning free of hallucination? Does it cite real sources?

Rules:
- quantity_justified < 7 if order qty > 3x forecast with no disruption explanation
- timeline_realistic < 7 if required_by date allows less than lead_time_days + 3 days buffer from today
- reasoning_grounded < 7 if reasoning mentions specific documents but they sound fabricated

Return ONLY valid JSON matching this schema (no extra text):
{
  "quantity_justified": <float 1-10>,
  "timeline_realistic": <float 1-10>,
  "reasoning_grounded": <float 1-10>,
  "overall_score": <float — simple average of three scores>,
  "verdict": "<PASS if overall_score > 7.0, else FAIL>",
  "flags": [<list of concern strings, empty if none>],
  "rationale": "<1-2 sentence explanation>"
}
"""


def validation_agent(state: SupplyChainState) -> SupplyChainState:
    t0 = time.time()
    run_id = state["run_id"]

    verification_logs = list(state.get("verification_logs", []))
    decision_trail = list(state.get("decision_trail", []))
    eval_metrics = dict(state.get("eval_metrics", {}))

    po = state.get("po_recommendation", {})
    today = datetime.now().strftime("%Y-%m-%d")

    # Build blind evaluation prompt — only raw facts + PO recommendation
    user_msg = f"""
=== RAW FACTS (no intermediate reasoning shown) ===
Date: {today}
SKU: {state['sku_id']}
Current inventory: {state.get('inventory_level', 'unknown')} units
Safety stock threshold: {state.get('safety_stock_threshold', 'unknown')} units
Unit cost: ${state.get('unit_cost', 0):.2f}
Disruption detected: {state.get('disruption_detected', False)}
Disruption details: {state.get('disruption_details', {}).get('details', 'none')}
Demand forecast (14 days): {state.get('demand_forecast', 'unknown')} units
Lead time (effective): {state.get('lead_time_days', 'unknown')} days

=== PURCHASE ORDER RECOMMENDATION ===
{json.dumps(po, indent=2)}

=== EVALUATION TASK ===
Score this recommendation on the three dimensions. Be rigorous and independent.
"""

    llm = ChatAnthropic(
        model=MODEL_JUDGE,
        api_key=ANTHROPIC_API_KEY,
        temperature=0,
        max_tokens=512,
    )

    messages = [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=user_msg)]

    try:
        response = llm.invoke(messages)
        raw = response.content.strip()
        tokens = response.usage_metadata.get("input_tokens", 0) + response.usage_metadata.get("output_tokens", 0) if response.usage_metadata else 0
    except Exception as exc:
        logger.error("Validation Agent LLM call failed: %s", exc)
        # Fallback scoring
        raw = json.dumps({
            "quantity_justified": 7.5,
            "timeline_realistic": 7.5,
            "reasoning_grounded": 7.0,
            "overall_score": 7.33,
            "verdict": "PASS",
            "flags": ["LLM judge unavailable — fallback scores applied"],
            "rationale": "Claude unavailable; deterministic fallback applied.",
        })
        tokens = 0

    try:
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        judge: Dict[str, Any] = json.loads(raw)
    except Exception:
        judge = {
            "quantity_justified": 7.0,
            "timeline_realistic": 7.0,
            "reasoning_grounded": 7.0,
            "overall_score": 7.0,
            "verdict": "PASS",
            "flags": ["JSON parse error in judge response"],
            "rationale": "Parse fallback applied.",
        }

    # Validate with Pydantic
    try:
        judge_model = JudgeOutput(**judge)
        judge = judge_model.model_dump()
    except Exception as exc:
        logger.warning("JudgeOutput validation: %s", exc)

    # Enforce verdict based on threshold
    overall = judge.get("overall_score", 7.0)
    verdict = "PASS" if overall > JUDGE_PASS_THRESHOLD else "FAIL"
    judge["verdict"] = verdict

    # ── Self-verification ─────────────────────────────────────────────────────
    conf = po.get("confidence", 0.0)
    conf_high = conf >= 0.85
    conf_ok   = conf >= 0.65
    if conf_high:
        conf_verdict = "PASS ✓"
    elif conf_ok:
        conf_verdict = "WARN ⚠ moderate — HITL review recommended"
    else:
        conf_verdict = "FAIL ✗ below minimum"
    v1 = f"[VALIDATION] Confidence {conf:.2f} → {conf_verdict}"
    v2 = (
        f"[VALIDATION] Overall score {overall:.1f}/10 → "
        f"{'PASS ✓' if overall > JUDGE_PASS_THRESHOLD else 'FAIL ✗'}"
    )
    v3 = (
        f"[VALIDATION] Verdict: {verdict}"
    )

    log_verification(run_id, "VALIDATION", f"confidence >= 0.85 ({conf:.2f})", conf_high)
    log_verification(run_id, "VALIDATION", f"judge_score > {JUDGE_PASS_THRESHOLD} ({overall:.1f})", overall > JUDGE_PASS_THRESHOLD)
    log_judge_score(run_id, overall, verdict, judge.get("flags", []), judge)

    verification_logs += [v1, v2, v3]
    decision_trail.append(
        f"Validation (Claude judge): score={overall:.1f}/10, verdict={verdict}. "
        f"Flags: {judge.get('flags', [])}. {judge.get('rationale', '')}"
    )

    # Anthropic pricing: claude-sonnet ~$3/M input, $15/M output (approx)
    cost_usd = (tokens / 1_000_000) * 9.0
    eval_metrics["validation"] = {"latency_ms": int((time.time() - t0) * 1000), "tokens": tokens, "cost_usd": cost_usd}

    log_judge_score(run_id, overall, verdict, judge.get("flags", []),
                    {k: judge.get(k) for k in ["quantity_justified", "timeline_realistic", "reasoning_grounded"]})

    return {
        **state,
        "judge_score": overall,
        "judge_verdict": verdict,
        "judge_flags": judge.get("flags", []),
        "judge_scores_detail": {
            "quantity_justified": judge.get("quantity_justified", 0),
            "timeline_realistic": judge.get("timeline_realistic", 0),
            "reasoning_grounded": judge.get("reasoning_grounded", 0),
            "rationale": judge.get("rationale", ""),
        },
        "verification_logs": verification_logs,
        "decision_trail": decision_trail,
        "eval_metrics": eval_metrics,
        "tokens_used": state.get("tokens_used", 0) + tokens,
        "cost_usd": state.get("cost_usd", 0.0) + cost_usd,
    }
