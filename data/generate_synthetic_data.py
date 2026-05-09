"""
Generate synthetic supply chain CSV data.

Run this script once to create data/kaggle/supply_chain.csv.
This mirrors the schema of the Kaggle dataset
(kaggle.com/datasets/prashantk93/supply-chain-management-for-car)
and fills gaps with synthetic but realistic values.

Column mapping from Kaggle → our schema
────────────────────────────────────────
Kaggle column          → Our field
───────────────────────────────────────
Product type           → category
SKU                    → sku_id
Price                  → unit_cost
Availability           → inventory_level
Number of products sold → units_sold
Revenue generated      → revenue
Stock levels           → safety_stock
Lead times             → lead_time_days
Order quantities       → order_quantity
Shipping times         → shipping_days
Supplier name          → supplier_id
Defect rates           → defect_rate
"""

import random
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

BASE_DIR = Path(__file__).parent
OUTPUT_PATH = BASE_DIR / "kaggle" / "supply_chain.csv"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

SKUS = [
    "SKU-4821", "SKU-1234", "SKU-9901", "SKU-7777",
    "SKU-2024", "SKU-5500", "SKU-3301", "SKU-8812",
    "SKU-6600", "SKU-4400",
]

SUPPLIERS = ["SUP-001", "SUP-002", "SUP-003", "SUP-004", "SUP-005"]

WAREHOUSES = ["WH-Detroit", "WH-Auburn_Hills", "WH-Windsor", "WH-Kokomo"]

CATEGORIES = {
    "SKU-4821": "Brake_Assembly",
    "SKU-1234": "Electrical_Harness",
    "SKU-9901": "Transmission_Module",
    "SKU-7777": "Seat_Foam",
    "SKU-2024": "ADAS_Sensor",
    "SKU-5500": "Fasteners",
    "SKU-3301": "Door_Panel",
    "SKU-8812": "Coolant_Pump",
    "SKU-6600": "Headlight_Assembly",
    "SKU-4400": "Exhaust_Component",
}

SUPPLIER_MAP = {
    "SKU-4821": "SUP-001",
    "SKU-1234": "SUP-002",
    "SKU-9901": "SUP-003",
    "SKU-7777": "SUP-001",
    "SKU-2024": "SUP-004",
    "SKU-5500": "SUP-005",
    "SKU-3301": "SUP-002",
    "SKU-8812": "SUP-003",
    "SKU-6600": "SUP-004",
    "SKU-4400": "SUP-005",
}

BASE_DEMAND = {
    "SKU-4821": 30, "SKU-1234": 55, "SKU-9901": 12,
    "SKU-7777": 8,  "SKU-2024": 5,  "SKU-5500": 120,
    "SKU-3301": 40, "SKU-8812": 18, "SKU-6600": 22, "SKU-4400": 35,
}

UNIT_COSTS = {
    "SKU-4821": 45.99, "SKU-1234": 12.50, "SKU-9901": 199.99,
    "SKU-7777": 89.00, "SKU-2024": 320.00, "SKU-5500": 8.75,
    "SKU-3301": 55.00, "SKU-8812": 110.50, "SKU-6600": 78.25, "SKU-4400": 42.00,
}

# Seasonal multipliers per quarter (Q1..Q4)
SEASONAL = {
    "SKU-4821": [0.85, 0.95, 1.00, 1.23],
    "SKU-9901": [0.90, 1.05, 0.95, 1.40],
    "SKU-2024": [1.05, 1.10, 1.00, 1.10],
}

# Supplier on-time delivery rates
ON_TIME_RATE = {
    "SUP-001": 0.94, "SUP-002": 0.61, "SUP-003": 0.75,
    "SUP-004": 0.89, "SUP-005": 0.88,
}

YEARS = [2022, 2023, 2024]
MONTHS = list(range(1, 13))

records = []
for year in YEARS:
    for month in MONTHS:
        for sku in SKUS:
            supplier = SUPPLIER_MAP[sku]
            q = (month - 1) // 3  # 0-indexed quarter
            seasonal = SEASONAL.get(sku, [1.0, 1.0, 1.0, 1.0])[q]
            base = BASE_DEMAND[sku]
            daily = int(base * seasonal * (1 + random.gauss(0, 0.08)))
            monthly_demand = daily * 30
            on_time = ON_TIME_RATE[supplier]
            # Lead time varies by disruption probability
            base_lt = {"SUP-001": 7, "SUP-002": 21, "SUP-003": 14, "SUP-004": 10, "SUP-005": 12}[supplier]
            actual_lt = max(1, int(base_lt * (1 + random.gauss(0, 0.15))))
            on_time_flag = int(random.random() < on_time)
            inventory = int(monthly_demand * 1.5 + random.gauss(0, 50))
            safety = int(monthly_demand * 0.5)
            order_qty = max(100, int(monthly_demand * 1.2))
            defect = round(random.uniform(0.005, 0.03), 4)

            records.append({
                "year": year,
                "month": month,
                "sku_id": sku,
                "category": CATEGORIES[sku],
                "warehouse": random.choice(WAREHOUSES),
                "supplier_id": supplier,
                "unit_cost": UNIT_COSTS[sku],
                "inventory_level": max(0, inventory),
                "safety_stock": safety,
                "monthly_demand": monthly_demand,
                "daily_demand": daily,
                "order_quantity": order_qty,
                "lead_time_days": actual_lt,
                "on_time_delivery": on_time_flag,
                "defect_rate": defect,
                "seasonal_factor": seasonal,
                "revenue": round(monthly_demand * UNIT_COSTS[sku], 2),
            })

df = pd.DataFrame(records)
df.to_csv(OUTPUT_PATH, index=False)
print(f"Generated {len(df)} rows → {OUTPUT_PATH}")
print(df.groupby("sku_id")[["monthly_demand", "on_time_delivery", "lead_time_days"]].mean().round(2))
