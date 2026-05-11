"""SOC2-compliant append-only JSONL audit trail.

Every tool call, verification check, agent completion, human decision,
and judge score is recorded here. The file is never truncated — only appended.
"""

import json
import fcntl
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from config import AUDIT_TRAIL_PATH


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_summary(obj: Any) -> str:
    """Lightweight content hash for integrity verification."""
    raw = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def log_event(
    run_id: str,
    agent: str,
    action: str,
    input_summary: Any,
    output_summary: Any,
    *,
    verification_passed: Optional[bool] = None,
    cost_usd: float = 0.0,
    tokens_used: int = 0,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Append one SOC2 audit record to the JSONL file (thread-safe)."""
    record: Dict[str, Any] = {
        "run_id": run_id,
        "timestamp": _now_iso(),
        "agent": agent,
        "action": action,
        "input_summary": input_summary,
        "output_summary": output_summary,
        "verification_passed": verification_passed,
        "cost_usd": round(cost_usd, 6),
        "tokens_used": tokens_used,
        "integrity_hash": _sha256_summary({"run_id": run_id, "agent": agent, "action": action}),
    }
    if extra:
        record.update(extra)

    AUDIT_TRAIL_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(AUDIT_TRAIL_PATH, "a", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            fh.write(json.dumps(record, default=str) + "\n")
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def log_tool_call(run_id: str, agent: str, tool_name: str, args: Dict, result: Any, cost_usd: float = 0.0, tokens: int = 0) -> None:
    log_event(run_id, agent, f"tool_call:{tool_name}", args, result, cost_usd=cost_usd, tokens_used=tokens)


def log_verification(run_id: str, agent: str, check: str, passed: bool, detail: str = "") -> None:
    log_event(run_id, agent, "verification", check, detail, verification_passed=passed)


def log_agent_complete(run_id: str, agent: str, output_summary: Any, cost_usd: float = 0.0, tokens: int = 0) -> None:
    log_event(run_id, agent, "agent_complete", None, output_summary, cost_usd=cost_usd, tokens_used=tokens)


def log_human_decision(
    run_id: str,
    hitl_tier: str,
    approved: bool,
    notes: str = "",
    escalation_tier: Optional[str] = None,
) -> None:
    """Log a human HITL decision. human_confirmed is always explicit — never auto-set."""
    log_event(
        run_id,
        "HITL_GATE",
        "human_decision",
        {"hitl_tier": hitl_tier, "escalation_tier": escalation_tier},
        {"human_confirmed": approved, "notes": notes},
        verification_passed=approved,
        extra={"human_confirmed": approved, "escalation_tier": escalation_tier},
    )


def log_judge_score(run_id: str, score: float, verdict: str, flags: list, detail: dict) -> None:
    log_event(run_id, "VALIDATION_AGENT", "judge_score", None, {"score": score, "verdict": verdict, "flags": flags, "detail": detail}, verification_passed=(verdict == "PASS"))


def read_trail(run_id: Optional[str] = None) -> list[dict]:
    """Return all audit records, optionally filtered by run_id."""
    if not AUDIT_TRAIL_PATH.exists():
        return []
    records = []
    with open(AUDIT_TRAIL_PATH, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if run_id is None or rec.get("run_id") == run_id:
                    records.append(rec)
            except json.JSONDecodeError:
                continue
    return records
