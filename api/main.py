"""
FastAPI service for Stellantis Supply Chain AI.

Endpoints:
  POST /run          — Start a new pipeline run
  POST /resume/{tid} — Resume SOFT/HARD HITL-paused pipeline
  GET  /status/{tid} — Get current state of a run
  GET  /audit        — Stream SOC2 audit trail
  GET  /health       — Health check
  GET  /docs         — Auto-generated OpenAPI (FastAPI built-in)
"""

import asyncio
import logging
import os
import uuid
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config import API_HOST, API_PORT

logger = logging.getLogger(__name__)

# Approver roles keyed by API key (loaded from env; format: "KEY1:role1,KEY2:role2")
_APPROVER_KEYS: Dict[str, str] = {}
_raw_keys = os.getenv("APPROVER_KEYS", "")
for pair in _raw_keys.split(","):
    if ":" in pair:
        k, role = pair.strip().split(":", 1)
        _APPROVER_KEYS[k.strip()] = role.strip()


def _verify_approver(x_approver_key: Optional[str] = Header(default=None)) -> str:
    """FastAPI dependency — validates X-Approver-Key and returns the approver role."""
    if not _APPROVER_KEYS:
        # No keys configured → open (development mode)
        return "dev"
    if not x_approver_key or x_approver_key not in _APPROVER_KEYS:
        raise HTTPException(status_code=403, detail="Invalid or missing X-Approver-Key header")
    return _APPROVER_KEYS[x_approver_key]


app = FastAPI(
    title="Stellantis Supply Chain AI",
    description="Multi-agent LLM pipeline for procurement decisions",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory run registry (replace with Redis in production)
_RUNS: Dict[str, Dict[str, Any]] = {}


# ── Request / Response models ─────────────────────────────────────────────────

class RunRequest(BaseModel):
    sku_id: str = Field(default="SKU-4821", description="SKU to analyze")
    warehouse: str = Field(default="WH-Detroit")
    supplier_id: str = Field(default="SUP-001")
    inventory_override: Optional[int] = Field(default=None, description="Override current inventory level")
    scenario: Optional[str] = Field(default="normal", description="normal|q4_spike|port_strike|critical_stockout")


class ResumeRequest(BaseModel):
    approved: bool
    comment: str = ""
    rejection_reason: str = ""


class RunResponse(BaseModel):
    run_id: str
    thread_id: str
    status: str
    hitl_tier: str
    message: str


# ── Scenario presets ──────────────────────────────────────────────────────────

_SCENARIO_OVERRIDES = {
    "normal":           {"sku_id": "SKU-0000", "supplier_id": "SUP-003", "inventory_override": 30},
    "high_value":       {"sku_id": "SKU-0047", "supplier_id": "SUP-002", "inventory_override": 8},
    "port_strike":      {"sku_id": "SKU-0026", "supplier_id": "SUP-002", "inventory_override": 5},
    "critical_stockout":{"sku_id": "SKU-0052", "supplier_id": "SUP-001", "inventory_override": 1},
    "q4_spike":         {"sku_id": "SKU-0000", "supplier_id": "SUP-003", "inventory_override": 40},
}


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "stellantis-supply-chain-ai"}


@app.post("/run", response_model=RunResponse)
async def start_run(req: RunRequest, background_tasks: BackgroundTasks):
    """Start a new supply chain analysis pipeline run."""
    thread_id = str(uuid.uuid4())

    # Apply scenario overrides
    overrides = _SCENARIO_OVERRIDES.get(req.scenario, {})
    sku_id = overrides.get("sku_id", req.sku_id)
    supplier_id = overrides.get("supplier_id", req.supplier_id)
    inventory_override = overrides.get("inventory_override", req.inventory_override)

    _RUNS[thread_id] = {"status": "running", "thread_id": thread_id}

    def run_sync():
        try:
            from graph import run_pipeline
            state, run_id = run_pipeline(
                sku_id=sku_id,
                warehouse=req.warehouse,
                supplier_id=supplier_id,
                thread_id=thread_id,
                inventory_override=inventory_override,
            )
            _RUNS[thread_id] = {
                "status": "awaiting_hitl" if state.get("hitl_tier") in ("SOFT", "HARD") and not state.get("human_approved") else "complete",
                "run_id": run_id,
                "thread_id": thread_id,
                "state": state,
            }
        except Exception as exc:
            logger.error("Pipeline run failed: %s", exc)
            _RUNS[thread_id] = {"status": "error", "error": str(exc)}

    background_tasks.add_task(run_sync)

    return RunResponse(
        run_id=thread_id,
        thread_id=thread_id,
        status="started",
        hitl_tier="unknown",
        message="Pipeline started. Poll /status/{thread_id} for results.",
    )


@app.get("/status/{thread_id}")
async def get_status(thread_id: str):
    """Get current run state."""
    if thread_id not in _RUNS:
        raise HTTPException(status_code=404, detail="Run not found")
    run = _RUNS[thread_id]
    state = run.get("state", {})
    return {
        "thread_id": thread_id,
        "status": run.get("status"),
        "hitl_tier": state.get("hitl_tier", "unknown"),
        "human_approved": state.get("human_approved", False),
        "judge_verdict": state.get("judge_verdict", ""),
        "judge_score": state.get("judge_score", 0.0),
        "cost_usd": state.get("cost_usd", 0.0),
        "tokens_used": state.get("tokens_used", 0),
        "executive_summary": state.get("executive_summary", ""),
        "po_recommendation": state.get("po_recommendation", {}),
        "verification_logs": state.get("verification_logs", []),
        "decision_trail": state.get("decision_trail", []),
        "eval_metrics": state.get("eval_metrics", {}),
        "error": run.get("error") or state.get("error"),
    }


@app.post("/resume/{thread_id}", response_model=RunResponse)
async def resume_run(
    thread_id: str,
    req: ResumeRequest,
    background_tasks: BackgroundTasks,
    approver_role: str = Depends(_verify_approver),
):
    """Resume a HITL-paused pipeline with an authenticated human decision."""
    if thread_id not in _RUNS:
        raise HTTPException(status_code=404, detail="Run not found")
    if _RUNS[thread_id].get("status") != "awaiting_hitl":
        raise HTTPException(status_code=400, detail="Run is not awaiting HITL approval")

    if not req.approved and not req.rejection_reason and not req.comment:
        raise HTTPException(status_code=422, detail="rejection_reason or comment is required when rejecting")

    def resume_sync():
        try:
            from graph import resume_pipeline
            state = resume_pipeline(
                thread_id,
                req.approved,
                notes=req.comment,
                rejection_reason=req.rejection_reason if not req.approved else "",
                approver_role=approver_role,
            )
            _RUNS[thread_id]["state"] = state
            _RUNS[thread_id]["status"] = "complete"
        except Exception as exc:
            logger.error("Resume failed: %s", exc)
            _RUNS[thread_id]["status"] = "error"
            _RUNS[thread_id]["error"] = str(exc)

    background_tasks.add_task(resume_sync)
    return RunResponse(
        run_id=thread_id,
        thread_id=thread_id,
        status="resuming",
        hitl_tier=_RUNS[thread_id].get("state", {}).get("hitl_tier", "unknown"),
        message=f"Pipeline resuming with decision: {'APPROVED' if req.approved else 'REJECTED'} (role: {approver_role})",
    )


@app.get("/audit")
async def get_audit_trail(run_id: Optional[str] = None, limit: int = 100):
    """Return SOC2 audit trail records."""
    from audit.soc2_logger import read_trail
    records = read_trail(run_id)
    return {"records": records[-limit:], "total": len(records)}


@app.get("/runs")
async def list_runs():
    """List all tracked runs and their statuses."""
    return [
        {
            "thread_id": tid,
            "status": info.get("status"),
            "hitl_tier": info.get("state", {}).get("hitl_tier"),
            "judge_verdict": info.get("state", {}).get("judge_verdict"),
            "cost_usd": info.get("state", {}).get("cost_usd", 0),
        }
        for tid, info in _RUNS.items()
    ]


@app.get("/metrics")
async def get_metrics():
    """Aggregate metrics across all completed runs."""
    completed = [r for r in _RUNS.values() if r.get("status") == "complete" and r.get("state")]
    if not completed:
        return {"message": "No completed runs yet."}
    costs = [r["state"].get("cost_usd", 0) for r in completed]
    scores = [r["state"].get("judge_score", 0) for r in completed if r["state"].get("judge_score")]
    tiers = [r["state"].get("hitl_tier", "AUTO") for r in completed]
    windows = [r["state"].get("decision_window_days", 0) for r in completed]
    return {
        "total_runs": len(completed),
        "avg_cost_usd": sum(costs) / len(costs) if costs else 0,
        "avg_judge_score": sum(scores) / len(scores) if scores else 0,
        "judge_pass_rate": sum(1 for r in completed if r["state"].get("judge_verdict") == "PASS") / len(completed),
        "hitl_trigger_rate": sum(1 for t in tiers if t != "AUTO") / len(tiers),
        "avg_decision_window_days": sum(windows) / len(windows) if windows else 0,
        "hitl_tier_breakdown": {
            "AUTO": tiers.count("AUTO"),
            "SOFT": tiers.count("SOFT"),
            "HARD": tiers.count("HARD"),
        },
    }


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run("api.main:app", host=API_HOST, port=API_PORT, reload=True)
