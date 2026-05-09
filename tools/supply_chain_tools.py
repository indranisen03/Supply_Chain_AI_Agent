"""
Supply-chain tools backed by real Kaggle data.

Dataset 1 (supply_chain_data.csv) drives:
  - get_inventory_level     → Availability, Stock levels, Price, Lead time
  - get_supplier_lead_time  → Lead time, defect_rate per SKU/supplier

Dataset 2 (DataCoSupplyChainDataset.csv) drives:
  - check_supplier_disruption → regional late_delivery_risk & on_time_rate
  - get_demand_forecast       → seasonal factors from real order history
  - run_scenario_model        → pure deterministic math (no LLM needed)

In production these call SAP S/4HANA APIs; the interfaces are identical.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Session-level cache — cleared between runs via clear_cache()
_TOOL_CACHE: Dict[str, Any] = {}


def _cache_key(*args) -> str:
    return ":".join(str(a) for a in args)


def _sku_catalog():
    """Lazy-load SKU catalog from Dataset 1. Cached via lru_cache in data_mapper."""
    from data.data_mapper import load_sku_catalog
    return load_sku_catalog()


def _order_history():
    """Lazy-load order history from Dataset 2. Cached via lru_cache in data_mapper."""
    from data.data_mapper import load_order_history
    return load_order_history()


# ── Tool implementations ──────────────────────────────────────────────────────

@tool
def get_inventory_level(sku_id: str, warehouse: str) -> Dict[str, Any]:
    """
    Return current inventory level, safety stock threshold, and unit cost
    for a SKU at a warehouse.
    Data source: supply_chain_data.csv (Dataset 1).
    """
    key = _cache_key("inventory", sku_id, warehouse)
    if key in _TOOL_CACHE:
        return _TOOL_CACHE[key]

    catalog = _sku_catalog()
    if not catalog.empty and sku_id in catalog.index:
        row = catalog.loc[sku_id]
        result = {
            "sku_id": sku_id,
            "warehouse": warehouse,
            "inventory_level": int(row["inventory_level"]),
            "safety_stock_threshold": int(row["safety_stock_threshold"]),
            "unit_cost": float(row["unit_cost"]),
            "supplier_id": str(row["supplier_id"]),
            "defect_rate": float(row["defect_rate"]),
            "transport_mode": str(row.get("transport_mode", "Standard")),
            "data_source": "supply_chain_data.csv",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    else:
        # Fallback for SKUs not in the catalog (e.g. custom demo IDs)
        logger.warning("SKU %s not in catalog — using fallback values", sku_id)
        result = {
            "sku_id": sku_id,
            "warehouse": warehouse,
            "inventory_level": 200,
            "safety_stock_threshold": 300,
            "unit_cost": 55.00,
            "supplier_id": "SUP-001",
            "defect_rate": 1.5,
            "transport_mode": "Standard Class",
            "data_source": "fallback",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    _TOOL_CACHE[key] = result
    return result


@tool
def check_supplier_disruption(supplier_id: str) -> Dict[str, Any]:
    """
    Check for active disruptions for a supplier.
    Disruption signal is derived from DataCo regional late_delivery_risk rates.
    Each SUP-00X maps to a representative shipping region.
    Data source: DataCoSupplyChainDataset.csv (Dataset 2).
    """
    key = _cache_key("disruption", supplier_id)
    if key in _TOOL_CACHE:
        return _TOOL_CACHE[key]

    # Map supplier IDs to DataCo regions (by on-time performance tier)
    # SUP-001 → best performers (Canada ~48%), SUP-005 → worst (Central Africa ~40%)
    _SUPPLIER_REGION_MAP = {
        "SUP-001": "Canada",
        "SUP-002": "Central America",   # high volume, avg performance
        "SUP-003": "Western Europe",    # high volume, below-avg
        "SUP-004": "Northern Europe",   # good performers
        "SUP-005": "Central Africa",    # lowest on-time rate
    }

    from data.data_mapper import get_supplier_performance
    region = _SUPPLIER_REGION_MAP.get(supplier_id, "Central America")
    perf_df = get_supplier_performance(region=region)

    if not perf_df.empty:
        row = perf_df[perf_df["region"] == region]
        if not row.empty:
            on_time = float(row.iloc[0]["on_time_rate"])
            late_risk = float(row.iloc[0]["late_risk_rate"])
            avg_lead = float(row.iloc[0]["avg_actual_lead"])
        else:
            on_time, late_risk, avg_lead = 0.43, 0.57, 3.5
    else:
        on_time, late_risk, avg_lead = 0.43, 0.57, 3.5

    # Determine disruption severity from on-time rate
    if on_time >= 0.46:
        severity, active = "none", False
        details = f"Operating normally. On-time rate {on_time:.1%} (region: {region})."
    elif on_time >= 0.43:
        severity, active = "LOW", True
        details = f"Minor delays. On-time rate {on_time:.1%} — ~{avg_lead:.1f}d avg lead time (region: {region})."
    elif on_time >= 0.40:
        severity, active = "MEDIUM", True
        details = f"Elevated late delivery risk {late_risk:.1%}. Expect +2-4 day delays (region: {region})."
    else:
        severity, active = "HIGH", True
        details = f"High disruption risk. On-time rate only {on_time:.1%}, late risk {late_risk:.1%} (region: {region})."

    result = {
        "supplier_id": supplier_id,
        "region": region,
        "disruption_active": active,
        "severity": severity,
        "details": details,
        "on_time_rate": round(on_time, 4),
        "late_delivery_risk_rate": round(late_risk, 4),
        "avg_actual_lead_time_days": round(avg_lead, 2),
        "data_source": "DataCoSupplyChainDataset.csv",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    _TOOL_CACHE[key] = result
    return result


@tool
def get_demand_forecast(sku_id: str, days: int) -> Dict[str, Any]:
    """
    Forecast total demand for a SKU over the next N days.
    Uses real seasonal factors computed from DataCo order history
    combined with the SKU's historical units_sold from Dataset 1.
    Data sources: both datasets.
    """
    key = _cache_key("forecast", sku_id, days)
    if key in _TOOL_CACHE:
        return _TOOL_CACHE[key]

    from data.data_mapper import get_seasonal_factors

    current_month = datetime.now().month
    seasonal = get_seasonal_factors(current_month)
    seasonal_factor = seasonal["seasonal_factor"]

    # Base daily demand from SKU catalog (units_sold / 30 days)
    catalog = _sku_catalog()
    if not catalog.empty and sku_id in catalog.index:
        units_sold = float(catalog.loc[sku_id].get("units_sold", 30))
        unit_cost = float(catalog.loc[sku_id]["unit_cost"])
    else:
        units_sold, unit_cost = 30.0, 55.0

    daily_demand = max(1, units_sold / 30.0)
    adjusted_daily = daily_demand * seasonal_factor
    total_forecast = int(adjusted_daily * days)

    # Confidence based on how much DataCo data covers this period
    order_hist = _order_history()
    confidence = 0.88 if not order_hist.empty else 0.72

    result = {
        "sku_id": sku_id,
        "forecast_days": days,
        "total_demand": total_forecast,
        "daily_demand": round(adjusted_daily, 2),
        "base_daily_demand": round(daily_demand, 2),
        "seasonal_factor": seasonal_factor,
        "seasonal_note": seasonal["note"],
        "confidence": confidence,
        "upper_bound": int(total_forecast * 1.15),
        "lower_bound": int(total_forecast * 0.85),
        "unit_cost": unit_cost,
        "data_source": "supply_chain_data.csv + DataCoSupplyChainDataset.csv",
    }

    _TOOL_CACHE[key] = result
    return result


@tool
def get_supplier_lead_time(supplier_id: str) -> Dict[str, Any]:
    """
    Return lead time and reliability for a supplier.
    Combines Dataset 1 SKU-level lead times with Dataset 2 actual delivery data.
    """
    key = _cache_key("lead_time", supplier_id)
    if key in _TOOL_CACHE:
        return _TOOL_CACHE[key]

    # Get actual average lead time from DataCo regional data
    from data.data_mapper import get_supplier_performance
    _SUPPLIER_REGION_MAP = {
        "SUP-001": "Canada",
        "SUP-002": "Central America",
        "SUP-003": "Western Europe",
        "SUP-004": "Northern Europe",
        "SUP-005": "Central Africa",
    }
    region = _SUPPLIER_REGION_MAP.get(supplier_id, "Central America")
    perf_df = get_supplier_performance(region=region)

    if not perf_df.empty:
        row = perf_df[perf_df["region"] == region]
        if not row.empty:
            avg_actual = float(row.iloc[0]["avg_actual_lead"])
            avg_sched = float(row.iloc[0]["avg_scheduled_lead"])
            on_time = float(row.iloc[0]["on_time_rate"])
        else:
            avg_actual, avg_sched, on_time = 3.5, 3.0, 0.43
    else:
        avg_actual, avg_sched, on_time = 3.5, 3.0, 0.43

    # Get contracted/catalog lead time from Dataset 1 for this supplier's SKUs
    catalog = _sku_catalog()
    sup_skus = catalog[catalog["supplier_id"] == supplier_id] if not catalog.empty else None
    catalog_lt = float(sup_skus["lead_time_days"].mean()) if sup_skus is not None and len(sup_skus) > 0 else 14

    # Effective lead time = weighted blend of catalog (contracted) and actual
    effective_lt = round(catalog_lt * 0.6 + avg_actual * 0.4, 1)
    std_dev = round(abs(avg_actual - avg_sched) + 0.5, 1)

    result = {
        "supplier_id": supplier_id,
        "region": region,
        "lead_time_days": int(effective_lt),
        "catalog_lead_time_days": round(catalog_lt, 1),
        "actual_avg_lead_time_days": round(avg_actual, 2),
        "scheduled_avg_lead_time_days": round(avg_sched, 2),
        "reliability_score": round(on_time, 4),
        "std_dev_days": std_dev,
        "worst_case_days": int(effective_lt + 2 * std_dev),
        "data_source": "supply_chain_data.csv + DataCoSupplyChainDataset.csv",
    }

    _TOOL_CACHE[key] = result
    return result


@tool
def run_scenario_model(
    sku_id: str,
    supplier_id: str,
    current_inventory: int,
    order_quantity: int,
    daily_demand: int,
    lead_time_days: int,
) -> Dict[str, Any]:
    """
    Run 3 what-if procurement scenarios and compute stockout risk.
    Pure deterministic math — no LLM or data file needed.
    """
    def stockout_prob(delay_days: int) -> float:
        days_to_arrival = delay_days + lead_time_days
        consumed = daily_demand * days_to_arrival
        if current_inventory >= consumed:
            return max(0.02, (consumed / max(1, current_inventory)) * 0.1)
        shortfall = consumed - current_inventory
        return min(0.99, round(shortfall / max(1, daily_demand * lead_time_days), 3))

    def days_until_50pct_risk() -> int:
        for d in range(1, 45):
            consumed = daily_demand * d
            if consumed >= current_inventory * 0.5:
                return d
        return 45

    scenarios = [
        {
            "scenario": "order_today",
            "delay_days": 0,
            "stockout_probability": stockout_prob(0),
            "units_at_arrival": max(0, current_inventory - daily_demand * lead_time_days),
        },
        {
            "scenario": "wait_7_days",
            "delay_days": 7,
            "stockout_probability": stockout_prob(7),
            "units_at_arrival": max(0, current_inventory - daily_demand * (lead_time_days + 7)),
        },
        {
            "scenario": "wait_14_days",
            "delay_days": 14,
            "stockout_probability": stockout_prob(14),
            "units_at_arrival": max(0, current_inventory - daily_demand * (lead_time_days + 14)),
        },
    ]

    best = min(scenarios, key=lambda s: s["stockout_probability"])

    return {
        "sku_id": sku_id,
        "supplier_id": supplier_id,
        "current_inventory": current_inventory,
        "daily_demand": daily_demand,
        "lead_time_days": lead_time_days,
        "scenarios": scenarios,
        "decision_window_days": days_until_50pct_risk(),
        "recommended_scenario": best["scenario"],
        "data_source": "deterministic_model",
    }


def clear_cache() -> None:
    """Clear the session tool cache between pipeline runs."""
    _TOOL_CACHE.clear()


ALL_TOOLS = [
    get_inventory_level,
    check_supplier_disruption,
    get_demand_forecast,
    get_supplier_lead_time,
    run_scenario_model,
]
