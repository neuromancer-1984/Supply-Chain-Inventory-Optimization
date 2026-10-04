"""Seed warehouses, suppliers, SKUs and the baseline reorder policy.

    python -m src.seed
"""
import numpy as np
import pandas as pd
from sqlalchemy import text

from . import config
from .db import copy_df, get_engine, load_pairs, set_meta
from .policy import baseline_policy


def main():
    rng = np.random.default_rng(config.SEED)
    eng = get_engine()
    with eng.begin() as c:
        c.execute(text("TRUNCATE forecasts, forecast_accuracy, forecast_runs, daily_kpis, stockout_events, "
                       "purchase_orders, inventory_snapshot, sales_daily, reorder_params, skus, suppliers, "
                       "warehouses, sim_meta RESTART IDENTITY CASCADE"))

    # warehouses
    wh = pd.DataFrame([(i + 1, n, c, f) for i, (n, c, f) in enumerate(config.WAREHOUSES)],
                      columns=["warehouse_id", "name", "city", "demand_factor"])
    copy_df(wh, "warehouses")

    # suppliers
    n = config.N_SUPPLIERS
    sup = pd.DataFrame({
        "supplier_id": np.arange(1, n + 1),
        "name": [f"Supplier {i:02d}" for i in range(1, n + 1)],
        "avg_lead_time_days": rng.integers(3, 15, n),
        "lead_time_std_days": np.round(rng.uniform(0.5, 2.5, n), 2),
        "delay_prob": np.round(rng.uniform(0.03, 0.20, n), 3),
    })
    copy_df(sup, "suppliers")

    # SKUs
    cats = list(config.CATEGORIES)
    shares = np.array([config.CATEGORIES[c][0] for c in cats])
    cat = rng.choice(cats, size=config.N_SKUS, p=shares / shares.sum())
    median_cost = np.array([config.CATEGORIES[c][1] for c in cat], float)
    cost = np.round(median_cost * rng.lognormal(0, 0.5, config.N_SKUS), 2)
    skus = pd.DataFrame({
        "sku_id": np.arange(1, config.N_SKUS + 1),
        "sku_code": [f"SKU-{i:05d}" for i in range(1, config.N_SKUS + 1)],
        "category": cat,
        "unit_cost": cost,
        "unit_price": np.round(cost * rng.uniform(1.2, 1.8, config.N_SKUS), 2),
        "holding_cost_pct": np.round(rng.uniform(0.12, 0.25, config.N_SKUS), 3),
        "supplier_id": rng.integers(1, n + 1, config.N_SKUS),
        "base_daily_demand": np.round(rng.lognormal(np.log(1.2), 0.9, config.N_SKUS), 3),
        "trend_pct_per_month": np.round(rng.uniform(-3, 6, config.N_SKUS), 2),
    })
    copy_df(skus, "skus")

    # baseline policy for every SKU x warehouse
    pairs = load_pairs()
    rop, S = baseline_policy(pairs)
    pol = pd.DataFrame({"sku_id": pairs["sku_id"], "warehouse_id": pairs["warehouse_id"],
                        "policy_name": config.BASELINE_POLICY, "reorder_point": rop, "order_up_to": S})
    copy_df(pol, "reorder_params")
    set_meta("active_policy", config.BASELINE_POLICY)
    print(f"Seeded {len(wh)} warehouses, {len(sup)} suppliers, {len(skus):,} SKUs, "
          f"{len(pairs):,} SKU-warehouse pairs, baseline policy.")


if __name__ == "__main__":
    main()
