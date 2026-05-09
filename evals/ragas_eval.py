"""
RAGAS + custom E2E evaluations for Stellantis Supply Chain AI.

Component evals (RAGAS):
  - faithfulness: RAG answer is grounded in retrieved context
  - context_precision: retrieved context is relevant

E2E objective evals:
  - qty_coverage: recommended_qty >= forecast + safety_stock
  - timeline_valid: required_by > today + lead_time_days
  - cost_per_run < $0.50

E2E subjective:
  - LLM judge scores (from state.judge_score)

Run: python -m evals.ragas_eval
"""

import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


# ── Objective E2E Evaluations ─────────────────────────────────────────────────

def eval_qty_coverage(state: Dict[str, Any]) -> Dict[str, Any]:
    """Check if recommended_qty >= demand_forecast + safety_stock."""
    po = state.get("po_recommendation", {})
    qty = po.get("quantity", 0)
    forecast = state.get("demand_forecast", 0)
    safety_stock = state.get("safety_stock_threshold", 0)
    required = forecast + safety_stock
    passed = qty >= required
    return {
        "eval": "qty_coverage",
        "passed": passed,
        "recommended_qty": qty,
        "required_qty": required,
        "shortfall": max(0, required - qty),
    }


def eval_timeline_valid(state: Dict[str, Any]) -> Dict[str, Any]:
    """Check if required_by > today + lead_time_days."""
    po = state.get("po_recommendation", {})
    required_by_str = po.get("required_by", "")
    lead_time = state.get("lead_time_days", 0)
    today = datetime.now()
    min_required_by = today + timedelta(days=lead_time)
    try:
        required_by = datetime.strptime(required_by_str, "%Y-%m-%d")
        passed = required_by > min_required_by
    except Exception:
        passed = False
        required_by = today
    return {
        "eval": "timeline_valid",
        "passed": passed,
        "required_by": required_by_str,
        "min_required_by": min_required_by.strftime("%Y-%m-%d"),
        "lead_time_days": lead_time,
    }


def eval_cost_per_run(state: Dict[str, Any]) -> Dict[str, Any]:
    """Check if cost_usd < $0.50."""
    cost = state.get("cost_usd", 0.0)
    passed = cost < 0.50
    return {
        "eval": "cost_per_run",
        "passed": passed,
        "cost_usd": cost,
        "cap_usd": 0.50,
        "overage_usd": max(0, cost - 0.50),
    }


def eval_judge_score(state: Dict[str, Any]) -> Dict[str, Any]:
    """Subjective LLM judge evaluation."""
    score = state.get("judge_score", 0.0)
    verdict = state.get("judge_verdict", "")
    passed = verdict == "PASS" and score > 7.0
    return {
        "eval": "judge_score",
        "passed": passed,
        "score": score,
        "verdict": verdict,
        "threshold": 7.0,
    }


def run_all_evals(state: Dict[str, Any]) -> Dict[str, Any]:
    """Run all E2E evals and return a summary report."""
    evals = [
        eval_qty_coverage(state),
        eval_timeline_valid(state),
        eval_cost_per_run(state),
        eval_judge_score(state),
    ]
    passed = sum(1 for e in evals if e["passed"])
    return {
        "run_id": state.get("run_id", "unknown"),
        "total_evals": len(evals),
        "passed": passed,
        "failed": len(evals) - passed,
        "pass_rate": passed / len(evals),
        "evals": evals,
    }


# ── RAGAS Component Evals ──────────────────────────────────────────────────────

def run_ragas_eval(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run RAGAS faithfulness + context_precision on the RAG-grounded reasoning.
    Falls back gracefully if RAGAS or its dependencies are unavailable.
    """
    po = state.get("po_recommendation", {})
    reasoning = po.get("reasoning", "")
    rag_sources = po.get("rag_sources", [])

    if not reasoning or not rag_sources:
        return {"ragas": "skipped", "reason": "No RAG reasoning available"}

    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, context_precision
        from datasets import Dataset

        sample = {
            "question": [f"Should we order {po.get('quantity')} units of {po.get('part_id')}?"],
            "answer": [reasoning],
            "contexts": [rag_sources],
            "ground_truth": ["Order quantity should cover forecast and safety stock per Stellantis policy."],
        }
        dataset = Dataset.from_dict(sample)
        result = evaluate(dataset, metrics=[faithfulness, context_precision])
        return {
            "ragas": "completed",
            "faithfulness": float(result["faithfulness"]),
            "context_precision": float(result["context_precision"]),
        }
    except ImportError:
        return {"ragas": "skipped", "reason": "ragas package not available"}
    except Exception as exc:
        logger.warning("RAGAS eval failed: %s", exc)
        return {"ragas": "error", "reason": str(exc)}


# ── CLI runner ─────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO)
    print("Running Stellantis Supply Chain AI Evals...")

    # Try to load last audit record as test state
    try:
        from config import AUDIT_TRAIL_PATH
        from audit.soc2_logger import read_trail
        records = read_trail()
        complete_runs = [r for r in records if r.get("action") == "run_complete"]
        if not complete_runs:
            print("No completed runs found in audit trail. Run demo.py first.")
            sys.exit(0)
        # Load state from API for last run
        import requests
        last_run = complete_runs[-1]
        run_id = last_run.get("run_id")
        resp = requests.get(f"http://localhost:8000/status/{run_id}", timeout=10)
        state = resp.json() if resp.ok else {}
    except Exception as e:
        print(f"Could not load run state: {e}")
        # Use mock state for eval testing
        state = {
            "run_id": "test-eval",
            "po_recommendation": {
                "part_id": "SKU-4821",
                "quantity": 800,
                "required_by": (datetime.now() + timedelta(days=21)).strftime("%Y-%m-%d"),
                "supplier_id": "SUP-001",
                "estimated_value": 36792.0,
                "confidence": 0.91,
                "reasoning": "Per Stellantis Responsible Purchasing Guidelines, order quantity covers 14-day forecast plus safety buffer.",
                "rag_sources": ["global_responsible_purchasing", "stellantis_code_of_conduct"],
            },
            "demand_forecast": 420,
            "safety_stock_threshold": 300,
            "lead_time_days": 7,
            "judge_score": 8.3,
            "judge_verdict": "PASS",
            "cost_usd": 0.18,
            "tokens_used": 3200,
        }

    print("\n=== E2E Objective Evaluations ===")
    report = run_all_evals(state)
    for e in report["evals"]:
        status = "✓ PASS" if e["passed"] else "✗ FAIL"
        print(f"  {status} — {e['eval']}: {json.dumps({k: v for k, v in e.items() if k not in ('eval', 'passed')})}")
    print(f"\n  Result: {report['passed']}/{report['total_evals']} passed ({report['pass_rate']:.1%})")

    print("\n=== RAGAS Component Evaluations ===")
    ragas_result = run_ragas_eval(state)
    print(f"  {json.dumps(ragas_result, indent=2)}")

    # Surface in output for Streamlit integration
    output = {"e2e": report, "ragas": ragas_result}
    output_path = Path(__file__).parent / "last_eval_result.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFull eval results written to: {output_path}")


if __name__ == "__main__":
    main()
