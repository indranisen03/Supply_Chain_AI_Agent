"""
Demo script — runs 3 scenarios and prints a full report.

Usage:
  python demo.py
  python demo.py --scenario port_strike
  python demo.py --all

Scenarios:
  1. normal       — SKU-4821 below safety stock, SUP-001 (healthy)   → SOFT or HARD
  2. high_value   — SKU-2024 (ADAS sensor, unit_cost=$320) → HARD block
  3. port_strike  — SUP-002 active disruption + low inventory        → HARD
"""

import argparse
import json
import logging
import sys
import time
import uuid
from datetime import datetime
from typing import Any, Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("demo")


SCENARIOS = {
    "normal": {
        "name": "Normal Reorder — SOFT Tier",
        # SKU-0000: unit_cost=$69.81, safety_stock=58, SUP-003 (Western Europe MEDIUM disruption)
        # Estimated PO: ~500 units × $69.81 = ~$34,905 → SOFT tier ($10k–$50k)
        "sku_id": "SKU-0000",
        "warehouse": "WH-Detroit",
        "supplier_id": "SUP-003",
        "inventory_override": 30,        # below safety_stock=58 → triggers reorder
        "description": (
            "SKU-0000 (unit_cost=$69.81) below safety stock. SUP-003 / Western Europe "
            "shows elevated late-risk (58.5%). Estimated PO ~$35k → SOFT HITL tier."
        ),
    },
    "high_value": {
        "name": "High-Value — HARD Block",
        # SKU-0047: unit_cost=$95.71, units_sold=910/mo (high demand), safety_stock=4
        # With SUP-002 disruption (MEDIUM) → lead time extends → large qty needed
        # Estimated PO: ~900 units × $95.71 = ~$86k → HARD tier (>$50k)
        "sku_id": "SKU-0047",
        "warehouse": "WH-Auburn_Hills",
        "supplier_id": "SUP-002",
        "inventory_override": 8,         # critically low against high demand
        "description": (
            "SKU-0047 (unit_cost=$95.71, high-volume 910 units/mo) critically low. "
            "SUP-002 / Central America MEDIUM disruption extends lead time. "
            "Estimated PO ~$86k → HARD HITL block."
        ),
    },
    "port_strike": {
        "name": "Critical Disruption — HARD Block",
        # SKU-0026: unit_cost=$97.45, inventory=9 (near-zero), SUP-002 disrupted
        # Stockout imminent — decision window < 1 day
        "sku_id": "SKU-0026",
        "warehouse": "WH-Windsor",
        "supplier_id": "SUP-002",
        "inventory_override": 5,         # near-zero stock
        "description": (
            "SKU-0026 (unit_cost=$97.45) near-zero inventory with SUP-002 active "
            "disruption (Central America, late-risk 57.1%). "
            "Critical stockout risk — decision window < 1 day → HARD HITL block."
        ),
    },
}


def print_separator(title: str = "") -> None:
    width = 70
    if title:
        print(f"\n{'═' * 10} {title} {'═' * max(0, width - len(title) - 12)}")
    else:
        print("─" * width)


def run_scenario(scenario_key: str, auto_approve: bool = True) -> Dict[str, Any]:
    """Execute a single demo scenario end-to-end."""
    cfg = SCENARIOS[scenario_key]
    print_separator(f"SCENARIO: {cfg['name']}")
    print(f"Description: {cfg['description']}")
    print(f"SKU: {cfg['sku_id']} | Supplier: {cfg['supplier_id']} | Inventory: {cfg['inventory_override']} units")
    print()

    start_time = time.time()

    # Import here to allow env var setup before import
    from graph import run_pipeline, resume_pipeline
    from tools.supply_chain_tools import clear_cache
    clear_cache()

    thread_id = str(uuid.uuid4())
    state, run_id = run_pipeline(
        sku_id=cfg["sku_id"],
        warehouse=cfg["warehouse"],
        supplier_id=cfg["supplier_id"],
        thread_id=thread_id,
        inventory_override=cfg["inventory_override"],
    )

    # Handle HITL tier
    tier = state.get("hitl_tier", "AUTO")
    print(f"\n[HITL] Tier: {tier}")
    if tier == "HARD" and not state.get("human_approved"):
        if auto_approve:
            print("[HITL] HARD tier — auto-approving for demo purposes...")
            state = resume_pipeline(thread_id, approved=True, comment="Demo auto-approval")
        else:
            print("[HITL] HARD tier — pipeline paused. Run resume_pipeline() to continue.")
            return {"state": state, "run_id": run_id, "paused": True}
    elif tier == "SOFT" and not state.get("human_approved"):
        print("[HITL] SOFT tier — auto-approving for demo purposes...")
        state = resume_pipeline(thread_id, approved=True, comment="Demo SOFT auto-approval")

    elapsed = time.time() - start_time

    # Print results
    print_separator("VERIFICATION LOGS")
    for log in state.get("verification_logs", []):
        icon = "✓" if "PASS" in log else "✗" if "FAIL" in log else "•"
        print(f"  {icon} {log}")

    print_separator("PO RECOMMENDATION")
    po = state.get("po_recommendation", {})
    if po:
        print(f"  Part:         {po.get('part_id')}")
        print(f"  Quantity:     {po.get('quantity'):,} units")
        print(f"  Supplier:     {po.get('supplier_id')}")
        print(f"  Required By:  {po.get('required_by')}")
        print(f"  Est. Value:   ${po.get('estimated_value', 0):,.2f}")
        print(f"  Confidence:   {po.get('confidence', 0):.1%}")
        print(f"  HITL Tier:    {po.get('hitl_tier')}")
        if po.get("rag_sources"):
            print(f"  RAG Sources:  {', '.join(po['rag_sources'])}")
        print(f"\n  Reasoning: {po.get('reasoning', 'N/A')[:300]}")

    print_separator("JUDGE VERDICT")
    print(f"  Score:   {state.get('judge_score', 0):.1f}/10")
    print(f"  Verdict: {state.get('judge_verdict', '—')}")
    detail = state.get("judge_scores_detail", {})
    if detail:
        for dim in ["quantity_justified", "timeline_realistic", "reasoning_grounded"]:
            print(f"  {dim}: {detail.get(dim, 0):.1f}/10")
    if state.get("judge_flags"):
        print(f"  Flags: {state['judge_flags']}")

    print_separator("EXECUTIVE SUMMARY")
    print(f"  {state.get('executive_summary', 'N/A')}")

    print_separator("METRICS")
    print(f"  Total cost:    ${state.get('cost_usd', 0):.4f}")
    print(f"  Total tokens:  {state.get('tokens_used', 0):,}")
    print(f"  Wall time:     {elapsed:.1f}s")
    print(f"  Decision window: {state.get('decision_window_days', 0)} days")
    print(f"  Human approved: {state.get('human_approved', False)}")

    metrics = state.get("eval_metrics", {})
    if metrics:
        print("\n  Per-agent breakdown:")
        for agent, m in metrics.items():
            print(f"    {agent:20s}: {m.get('latency_ms', 0):4d}ms | {m.get('tokens', 0):4d} tokens | ${m.get('cost_usd', 0):.4f}")

    return {"state": state, "run_id": run_id, "elapsed": elapsed}


def run_e2e_evals(results: list) -> None:
    """Run and print E2E evaluations for all completed scenarios."""
    from evals.ragas_eval import run_all_evals
    print_separator("E2E EVALUATIONS")
    for r in results:
        if not r.get("state"):
            continue
        report = run_all_evals(r["state"])
        print(f"\n  Scenario {r.get('run_id', 'unknown')[:8]}:")
        for e in report["evals"]:
            status = "✓ PASS" if e["passed"] else "✗ FAIL"
            print(f"    {status} — {e['eval']}")
        print(f"  Pass rate: {report['pass_rate']:.1%}")


def main():
    parser = argparse.ArgumentParser(description="Stellantis Supply Chain AI Demo")
    parser.add_argument("--scenario", choices=list(SCENARIOS.keys()), default=None)
    parser.add_argument("--all", action="store_true", help="Run all 3 scenarios")
    parser.add_argument("--no-auto-approve", action="store_true", help="Don't auto-approve HITL gates")
    args = parser.parse_args()

    auto_approve = not args.no_auto_approve

    # Generate synthetic data if needed
    from config import KAGGLE_CSV_PATH
    if not KAGGLE_CSV_PATH.exists():
        print("Generating synthetic supply chain data...")
        import subprocess
        subprocess.run([sys.executable, "data/generate_synthetic_data.py"], check=True)
        print("✓ Synthetic data ready\n")

    results = []
    if args.all or not args.scenario:
        for key in SCENARIOS:
            try:
                r = run_scenario(key, auto_approve=auto_approve)
                r["scenario"] = key
                results.append(r)
            except Exception as exc:
                logger.error("Scenario %s failed: %s", key, exc)
    else:
        r = run_scenario(args.scenario, auto_approve=auto_approve)
        r["scenario"] = args.scenario
        results.append(r)

    if results:
        run_e2e_evals(results)
        print_separator("DEMO COMPLETE")
        print(f"  Ran {len(results)} scenario(s)")
        print(f"  Audit trail: audit/audit_trail.jsonl")
        print(f"\n  Start the full UI:")
        print(f"    uvicorn api.main:app --reload &")
        print(f"    streamlit run ui/dashboard.py")


if __name__ == "__main__":
    main()
