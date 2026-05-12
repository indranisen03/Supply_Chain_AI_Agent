"""Procedural memory for HITL patterns — feeds dynamic thresholds into policy.py."""

import json
import fcntl
from pathlib import Path
from typing import Any, Dict, Optional

_MEMORY_FILE = Path(__file__).parent / "hitl_memory.json"

# How many recent runs to consider for dynamic threshold adjustments
_LOOKBACK = 20


def _load() -> Dict[str, Any]:
    if not _MEMORY_FILE.exists():
        return {"decisions": [], "thresholds": {}}
    with open(_MEMORY_FILE, "r", encoding="utf-8") as fh:
        try:
            return json.load(fh)
        except json.JSONDecodeError:
            return {"decisions": [], "thresholds": {}}


def _save(data: Dict[str, Any]) -> None:
    _MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_MEMORY_FILE, "w", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            json.dump(data, fh, indent=2, default=str)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def load_hitl_patterns() -> Dict[str, Any]:
    return _load()


def record_hitl_decision(
    run_id: str,
    tier: str,
    approved: bool,
    po_value: float,
    judge_score: float,
    rejection_reason: Optional[str] = None,
) -> None:
    """Append one HITL decision and recompute dynamic thresholds."""
    mem = _load()
    decisions: list = mem.setdefault("decisions", [])
    decisions.append({
        "run_id": run_id,
        "tier": tier,
        "approved": approved,
        "po_value": po_value,
        "judge_score": judge_score,
        "rejection_reason": rejection_reason,
    })
    # Keep only the last _LOOKBACK decisions
    mem["decisions"] = decisions[-_LOOKBACK:]
    mem["thresholds"] = _compute_thresholds(mem["decisions"])
    _save(mem)


def _compute_thresholds(decisions: list) -> Dict[str, Any]:
    """
    Adjust AUTO/SOFT caps based on recent approval patterns.
    If AUTO rejections > 30% of recent AUTO decisions, lower the AUTO cap by 10%.
    If SOFT approvals > 80%, raise the AUTO cap by 10%.
    """
    from config import HITL_AUTO_MAX_USD, HITL_SOFT_MAX_USD

    auto_decisions = [d for d in decisions if d["tier"] == "AUTO"]
    soft_decisions = [d for d in decisions if d["tier"] == "SOFT"]

    auto_cap = HITL_AUTO_MAX_USD
    soft_cap = HITL_SOFT_MAX_USD

    if len(auto_decisions) >= 5:
        auto_reject_rate = sum(1 for d in auto_decisions if not d["approved"]) / len(auto_decisions)
        if auto_reject_rate > 0.30:
            auto_cap = round(HITL_AUTO_MAX_USD * 0.90, 2)
        elif auto_reject_rate < 0.10 and len(auto_decisions) >= 10:
            auto_cap = round(min(HITL_AUTO_MAX_USD * 1.10, HITL_SOFT_MAX_USD * 0.9), 2)

    if len(soft_decisions) >= 5:
        soft_approve_rate = sum(1 for d in soft_decisions if d["approved"]) / len(soft_decisions)
        if soft_approve_rate > 0.80:
            soft_cap = round(HITL_SOFT_MAX_USD * 1.10, 2)

    return {"auto_cap_usd": auto_cap, "soft_cap_usd": soft_cap}


def get_dynamic_thresholds() -> Dict[str, float]:
    """Return currently computed dynamic HITL thresholds, or config defaults."""
    from config import HITL_AUTO_MAX_USD, HITL_SOFT_MAX_USD
    mem = _load()
    thresholds = mem.get("thresholds", {})
    return {
        "auto_cap_usd": thresholds.get("auto_cap_usd", HITL_AUTO_MAX_USD),
        "soft_cap_usd": thresholds.get("soft_cap_usd", HITL_SOFT_MAX_USD),
    }
