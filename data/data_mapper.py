"""
Data mapping layer — transforms both Kaggle datasets into the
unified supply chain schema used by all agents.

Dataset 1: supply_chain_data.csv  (harshsingh2209/supply-chain-analysis)
  100 rows | SKU-level catalog — inventory, pricing, lead times, defect rates

Dataset 2: DataCoSupplyChainDataset.csv  (shashwatwork/dataco-smart-supply-chain)
  180,519 rows | Order history 2015-2017 — demand trends, delivery performance

Column mapping
───────────────────────────────────────────────────────────────────────────────
Dataset 1                    → Our schema field
───────────────────────────────────────────────────────────────────────────────
SKU                          → sku_id        (normalised: "SKU0" → "SKU-0000")
Price                        → unit_cost
Availability                 → inventory_level
Stock levels                 → safety_stock_threshold
Lead times                   → shipping_lead_time_days
Lead time                    → manufacturing_lead_time_days
Order quantities             → typical_order_qty
Number of products sold      → units_sold
Supplier name                → supplier_id   (normalised: "Supplier 3" → "SUP-003")
Defect rates                 → defect_rate
Revenue generated            → revenue
Production volumes           → production_volume
Manufacturing costs          → manufacturing_cost
Shipping costs               → shipping_cost
Inspection results           → inspection_result
Transportation modes         → transport_mode

Dataset 2                    → Our schema field
───────────────────────────────────────────────────────────────────────────────
order date (DateOrders)      → order_date
Product Name                 → product_name
Order Item Quantity          → quantity_ordered
Order Item Product Price     → unit_price
Sales                        → sales_amount
Order Profit Per Order       → profit
Delivery Status              → delivery_status → on_time_delivery (bool)
Late_delivery_risk           → late_delivery_risk (0/1)
Days for shipping (real)     → actual_lead_time_days
Days for shipment (scheduled)→ scheduled_lead_time_days
Order Region                 → region
Market                       → market
Shipping Mode                → shipping_mode
Order Status                 → order_status
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ── File paths ────────────────────────────────────────────────────────────────
_BASE = Path(__file__).parent / "kaggle"
DS1_PATH = _BASE / "supply_chain_data.csv"
DS2_PATH = _BASE / "DataCoSupplyChainDataset.csv"

# ── Normalisation helpers ─────────────────────────────────────────────────────

def _norm_sku(raw: str) -> str:
    """'SKU0' → 'SKU-0000', 'SKU42' → 'SKU-0042'"""
    num = raw.replace("SKU", "").strip()
    try:
        return f"SKU-{int(num):04d}"
    except ValueError:
        return raw


def _norm_supplier(raw: str) -> str:
    """'Supplier 3' → 'SUP-003'"""
    num = raw.replace("Supplier", "").strip()
    try:
        return f"SUP-{int(num):03d}"
    except ValueError:
        return raw


def _on_time_flag(delivery_status: str) -> int:
    """Map DataCo Delivery Status → 1 (on-time) / 0 (late) / -1 (excluded)."""
    s = str(delivery_status).strip().lower()
    if s in ("advance shipping", "shipping on time"):
        return 1
    elif s == "late delivery":
        return 0
    else:  # Shipping canceled
        return -1


# ── Dataset 1 loader ──────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def load_sku_catalog() -> pd.DataFrame:
    """
    Load Dataset 1 as a normalised SKU catalog.
    Cached — loaded once per process.
    """
    if not DS1_PATH.exists():
        logger.warning("DS1 not found at %s", DS1_PATH)
        return pd.DataFrame()

    df = pd.read_csv(DS1_PATH, encoding="utf-8")

    df = df.rename(columns={
        "SKU": "sku_id",
        "Price": "unit_cost",
        "Availability": "inventory_level",
        "Stock levels": "safety_stock_threshold",
        "Lead times": "shipping_lead_time_days",
        "Lead time": "manufacturing_lead_time_days",
        "Order quantities": "typical_order_qty",
        "Number of products sold": "units_sold",
        "Supplier name": "supplier_id",
        "Defect rates": "defect_rate",
        "Revenue generated": "revenue",
        "Production volumes": "production_volume",
        "Manufacturing costs": "manufacturing_cost",
        "Shipping costs": "shipping_cost",
        "Inspection results": "inspection_result",
        "Transportation modes": "transport_mode",
        "Product type": "product_type",
        "Customer demographics": "customer_segment",
    })

    df["sku_id"] = df["sku_id"].apply(_norm_sku)
    df["supplier_id"] = df["supplier_id"].apply(_norm_supplier)

    # Effective lead time = max(shipping, manufacturing)
    df["lead_time_days"] = df[["shipping_lead_time_days", "manufacturing_lead_time_days"]].max(axis=1)

    logger.info("SKU catalog loaded: %d SKUs", len(df))
    return df.set_index("sku_id")


# ── Dataset 2 loader ──────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def load_order_history() -> pd.DataFrame:
    """
    Load Dataset 2 as a normalised order history table.
    Parses dates, derives on_time_delivery, extracts month/year/quarter.
    Cached — loaded once per process.
    """
    if not DS2_PATH.exists():
        logger.warning("DS2 not found at %s", DS2_PATH)
        return pd.DataFrame()

    df = pd.read_csv(DS2_PATH, encoding="latin-1", low_memory=False)

    df = df.rename(columns={
        "order date (DateOrders)": "order_date",
        "Product Name": "product_name",
        "Order Item Quantity": "quantity_ordered",
        "Order Item Product Price": "unit_price",
        "Sales": "sales_amount",
        "Order Profit Per Order": "profit",
        "Delivery Status": "delivery_status",
        "Late_delivery_risk": "late_delivery_risk",
        "Days for shipping (real)": "actual_lead_time_days",
        "Days for shipment (scheduled)": "scheduled_lead_time_days",
        "Order Region": "region",
        "Market": "market",
        "Shipping Mode": "shipping_mode",
        "Order Status": "order_status",
        "Order Item Id": "order_item_id",
        "Order Id": "order_id",
        "Customer Segment": "customer_segment",
        "Category Name": "category",
        "Department Name": "department",
        "Product Card Id": "product_id",
    })

    # Parse dates
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df = df.dropna(subset=["order_date"])
    df["year"] = df["order_date"].dt.year
    df["month"] = df["order_date"].dt.month
    df["quarter"] = df["order_date"].dt.quarter
    df["year_month"] = df["order_date"].dt.to_period("M")

    # On-time flag (exclude canceled)
    df["on_time_delivery"] = df["delivery_status"].apply(_on_time_flag)
    df = df[df["on_time_delivery"] != -1]  # remove canceled

    # Normalise product_name (strip whitespace)
    df["product_name"] = df["product_name"].str.strip()

    logger.info("Order history loaded: %d orders (%d–%d)", len(df),
                df["year"].min(), df["year"].max())
    return df


# ── Derived analytics for Historical Trend Agent ──────────────────────────────

def get_monthly_demand_trend(product_id: Optional[str] = None,
                              region: Optional[str] = None) -> pd.DataFrame:
    """
    Return monthly total quantity ordered, grouped by year+month.
    Optionally filtered by product_id or region.
    Used by Historical Trend Agent for demand delta calculation.
    """
    df = load_order_history()
    if df.empty:
        return pd.DataFrame()

    mask = pd.Series([True] * len(df), index=df.index)
    if product_id:
        mask &= df["product_id"].astype(str) == str(product_id)
    if region:
        mask &= df["region"] == region

    agg = (df[mask]
           .groupby(["year", "month", "quarter"])
           .agg(
               total_qty=("quantity_ordered", "sum"),
               total_sales=("sales_amount", "sum"),
               avg_lead_time=("actual_lead_time_days", "mean"),
               on_time_rate=("on_time_delivery", "mean"),
               late_risk_rate=("late_delivery_risk", "mean"),
               order_count=("order_id", "nunique"),
           )
           .reset_index()
           .sort_values(["year", "month"]))
    return agg


def get_supplier_performance(region: Optional[str] = None) -> pd.DataFrame:
    """
    Compute on-time delivery rate and average lead time by region/market.
    Since DataCo doesn't have explicit supplier IDs, we use region as a
    proxy for supplier routing (each region = one primary logistics channel).
    """
    df = load_order_history()
    if df.empty:
        return pd.DataFrame()

    mask = pd.Series([True] * len(df), index=df.index)
    if region:
        mask &= df["region"] == region

    agg = (df[mask]
           .groupby("region")
           .agg(
               on_time_rate=("on_time_delivery", "mean"),
               avg_actual_lead=("actual_lead_time_days", "mean"),
               avg_scheduled_lead=("scheduled_lead_time_days", "mean"),
               total_orders=("order_id", "nunique"),
               late_risk_rate=("late_delivery_risk", "mean"),
           )
           .reset_index())
    return agg


def get_sku_info(sku_id: str) -> Dict:
    """Look up a SKU in the catalog. Returns a dict of fields."""
    catalog = load_sku_catalog()
    if catalog.empty or sku_id not in catalog.index:
        return {}
    row = catalog.loc[sku_id]
    return row.to_dict()


def get_seasonal_factors(target_month: int) -> Dict:
    """
    Compute seasonal demand factor for a given month vs. annual average,
    using the full DataCo order history.
    """
    df = load_order_history()
    if df.empty:
        return {"seasonal_factor": 1.0, "note": "No data"}

    monthly_avg = df.groupby("month")["quantity_ordered"].mean()
    overall_avg = df["quantity_ordered"].mean()

    factor = monthly_avg.get(target_month, overall_avg) / overall_avg if overall_avg > 0 else 1.0
    quarter = (target_month - 1) // 3 + 1
    q_months = [(quarter - 1) * 3 + i for i in range(1, 4)]
    q_avg = df[df["month"].isin(q_months)]["quantity_ordered"].mean()
    q_factor = q_avg / overall_avg if overall_avg > 0 else 1.0

    return {
        "month": target_month,
        "seasonal_factor": round(float(factor), 3),
        "quarter": quarter,
        "quarter_factor": round(float(q_factor), 3),
        "is_peak_quarter": q_factor > 1.1,
        "note": f"Month {target_month} demand is {factor:.1%} of annual average; Q{quarter} factor={q_factor:.2f}",
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("\n--- SKU Catalog (first 5) ---")
    cat = load_sku_catalog()
    print(cat[["unit_cost", "inventory_level", "safety_stock_threshold",
               "lead_time_days", "supplier_id", "defect_rate"]].head().to_string())

    print("\n--- Order History sample ---")
    hist = load_order_history()
    print(hist[["year", "month", "product_name", "quantity_ordered",
                "on_time_delivery", "actual_lead_time_days"]].head(5).to_string())

    print("\n--- Monthly Demand Trend (all products) ---")
    trend = get_monthly_demand_trend()
    print(trend.tail(12).to_string(index=False))

    print("\n--- Supplier (region) Performance ---")
    perf = get_supplier_performance()
    print(perf.sort_values("on_time_rate", ascending=False).to_string(index=False))

    print("\n--- Seasonal Factors ---")
    for m in [1, 4, 7, 10, 11, 12]:
        s = get_seasonal_factors(m)
        print(f"  Month {m:2d}: factor={s['seasonal_factor']:.3f}  {s['note']}")
