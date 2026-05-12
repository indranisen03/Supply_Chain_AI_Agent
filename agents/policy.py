"""
Centralised procurement policy rules.

Extracted from optimization.py so that graph.py and validation.py
can share the same tier logic without circular imports.
"""

from typing import Tuple

# ── Role definitions ──────────────────────────────────────────────────────────

# Which HITL tiers each role is authorised to approve or reject.
# Roles are additive upward — a Director can act on everything below HARD too.
ROLE_TIER_PERMISSIONS: dict[str, list[str]] = {
    "analyst":       [],                          # view-only
    "coordinator":   ["AUTO"],
    "sr_manager_l5": ["AUTO", "SOFT"],
    "director_l6":   ["AUTO", "SOFT", "HARD"],
}

ROLE_DISPLAY_NAMES: dict[str, str] = {
    "analyst":       "Analyst",
    "coordinator":   "Coordinator",
    "sr_manager_l5": "Sr. Manager (L5)",
    "director_l6":   "Director (L6)",
}

# Minimum role required per tier (for error messages)
_TIER_MIN_ROLE: dict[str, str] = {
    "AUTO": "Coordinator",
    "SOFT": "Sr. Manager (L5)",
    "HARD": "Director (L6)",
}


def role_can_approve(role: str, tier: str) -> bool:
    """Return True if the role is authorised to approve/reject this tier."""
    return tier in ROLE_TIER_PERMISSIONS.get(role, [])


def minimum_role_for_tier(tier: str) -> str:
    """Human-readable minimum role name needed to act on a tier."""
    return _TIER_MIN_ROLE.get(tier, "Director (L6)")


# ── Per-agent retry caps (0 = no retries, escalate immediately) ──────────────
AGENT_MAX_RETRIES: dict[str, int] = {
    "SENSING": 2,
    "HISTORICAL_TREND": 2,
    "SIMULATION": 2,
    "OPTIMIZATION": 1,
    "VALIDATION": 0,   # escalate to HARD on any failure
    "SUMMARIZER": 3,
}

# Judge score below which AUTO → SOFT escalation is triggered
JUDGE_ESCALATION_THRESHOLD = 7.0


def determine_hitl_tier(estimated_value: float) -> Tuple[str, bool]:
    """
    Return (tier, requires_human_approval).
    All tiers require explicit human confirmation — no tier auto-approves.
    Uses dynamic thresholds when HITL memory has enough data.
    """
    from memory.hitl_memory import get_dynamic_thresholds
    thresholds = get_dynamic_thresholds()
    auto_cap = thresholds["auto_cap_usd"]
    soft_cap = thresholds["soft_cap_usd"]

    if estimated_value < auto_cap:
        return "AUTO", True
    elif estimated_value <= soft_cap:
        return "SOFT", True
    else:
        return "HARD", True


def should_escalate_to_soft(current_tier: str, judge_score: float) -> bool:
    """
    Return True when a judge score on an AUTO recommendation is low enough
    that a human should do a proper SOFT review.
    SOFT and HARD are already strict — no escalation needed for them here.
    """
    return current_tier == "AUTO" and judge_score < JUDGE_ESCALATION_THRESHOLD
