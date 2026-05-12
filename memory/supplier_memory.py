"""Procedural memory for supplier performance — persisted across runs."""

import json
import fcntl
from pathlib import Path
from typing import Any, Dict

_MEMORY_FILE = Path(__file__).parent / "supplier_memory.json"


def _load() -> Dict[str, Any]:
    if not _MEMORY_FILE.exists():
        return {}
    with open(_MEMORY_FILE, "r", encoding="utf-8") as fh:
        try:
            return json.load(fh)
        except json.JSONDecodeError:
            return {}


def _save(data: Dict[str, Any]) -> None:
    _MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_MEMORY_FILE, "w", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            json.dump(data, fh, indent=2, default=str)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def load_supplier_memory() -> Dict[str, Any]:
    """Return full supplier memory dict keyed by supplier_id."""
    return _load()


def update_supplier_memory(
    supplier_id: str,
    run_id: str,
    on_time: bool,
    lead_time_days: int,
    disruption_detected: bool,
    po_value: float,
) -> None:
    """Record one run's supplier outcome and update rolling stats."""
    mem = _load()
    entry = mem.setdefault(supplier_id, {
        "runs": 0,
        "on_time_count": 0,
        "total_lead_time": 0,
        "disruption_count": 0,
        "total_po_value": 0.0,
        "last_run_id": None,
    })
    entry["runs"] += 1
    if on_time:
        entry["on_time_count"] += 1
    entry["total_lead_time"] += lead_time_days
    if disruption_detected:
        entry["disruption_count"] += 1
    entry["total_po_value"] += po_value
    entry["last_run_id"] = run_id
    _save(mem)


def get_supplier_context(supplier_id: str) -> Dict[str, Any]:
    """Return derived stats for a supplier — used to enrich Optimization prompt."""
    mem = _load()
    entry = mem.get(supplier_id)
    if not entry or entry["runs"] == 0:
        return {"supplier_id": supplier_id, "history": "no prior runs"}
    runs = entry["runs"]
    return {
        "supplier_id": supplier_id,
        "total_runs": runs,
        "on_time_rate": round(entry["on_time_count"] / runs, 3),
        "avg_lead_time_days": round(entry["total_lead_time"] / runs, 1),
        "disruption_rate": round(entry["disruption_count"] / runs, 3),
        "avg_po_value_usd": round(entry["total_po_value"] / runs, 2),
    }
