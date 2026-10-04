"""Reorder-policy optimizer.

    python -m src.optimizer --service-level 0.95            # write policy to reorder_params
    python -m src.optimizer --service-level 0.95 --activate # ...and make the daily sim use it

Also exposes `sweep()` for the dashboard's what-if page: simulate baseline vs optimized
policies at several service levels and find the cost-minimising one.
"""
import argparse
from datetime import date

import numpy as np
import pandas as pd
from sqlalchemy import text

from . import config
from .db import copy_df, get_engine, load_pairs, pair_lookup, set_meta
from .engine import Engine
from .policy import baseline_policy, optimized_policy


def planning_inputs(pairs):
    """Forecast mean demand and forecast-error RMSE per pair (from the SQL views)."""
    q = pd.read_sql("SELECT m.sku_id, m.warehouse_id, m.mu, e.rmse FROM v_latest_forecast_mu m "
                    "JOIN v_pair_champion_error e USING (sku_id, warehouse_id)", get_engine())
    if q.empty:
        raise SystemExit("No forecasts yet. Run: python -m src.forecast_arena all")
    q = q.set_index(["sku_id", "warehouse_id"]).reindex(pair_lookup(pairs))
    mu = q["mu"].fillna(0).to_numpy(float)
    rmse = q["rmse"].fillna(q["rmse"].mean()).to_numpy(float)
    return mu, rmse


def evaluate(sample, rop, S, days=60, warmup=14, lt_mult=1.0, seed=7):
    """Run the engine in memory and return aggregate KPIs (excluding warm-up)."""
    start = date(2026, 1, 1)
    eng = Engine(sample, rop, S, on_hand=S, pipe=np.zeros((len(sample), config.MAX_LEAD), dtype=np.int64),
                 origin_date=start, seed=seed, lt_mult=lt_mult)
    agg = dict(demand=0, sold=0, inv=0.0, hold=0.0, lost_margin=0.0, orders=0)
    n_days = 0
    for i in range(days):
        k = eng.step(date.fromordinal(start.toordinal() + i))["kpi"]
        if i < warmup:
            continue
        n_days += 1
        agg["demand"] += k["units_demanded"]; agg["sold"] += k["units_sold"]
        agg["inv"] += k["inventory_value"]; agg["hold"] += k["holding_cost"]
        agg["lost_margin"] += k["lost_margin"]; agg["orders"] += k["orders_placed"]
    return dict(fill_rate=agg["sold"] / max(agg["demand"], 1), avg_inventory=agg["inv"] / n_days,
                holding_cost=agg["hold"], lost_margin=agg["lost_margin"],
                orders_per_day=agg["orders"] / n_days)


def make_sample(pairs, mu, rmse, n=3000, seed=1):
    """Sample of pairs in which the true mean demand equals the forecast mean (unbiased planning view)."""
    ok = np.nonzero(mu > 0.05)[0]
    sel = np.random.default_rng(seed).choice(ok, size=min(n, len(ok)), replace=False)
    s = pairs.iloc[sel].reset_index(drop=True).copy()
    s["base_mu"] = mu[sel]
    s["trend_pct"] = 0.0
    return s, mu[sel], rmse[sel]


def sweep(pairs, mu, rmse, levels=(0.80, 0.85, 0.90, 0.95, 0.97, 0.99, 0.995), lt_mult=1.0,
          hold_mult=1.0, lost_mult=1.0, n=3000, days=60):
    sample, m, r = make_sample(pairs, mu, rmse, n=n)
    rows = []
    rop, S = baseline_policy(sample, lt_mult=lt_mult)
    rows.append(dict(policy="baseline", service_level=np.nan, **evaluate(sample, rop, S, days, lt_mult=lt_mult)))
    for sl in levels:
        rop, S = optimized_policy(sample, m, r, sl, lt_mult=lt_mult)
        rows.append(dict(policy=f"optimized_{int(sl * 100)}", service_level=sl,
                         **evaluate(sample, rop, S, days, lt_mult=lt_mult)))
    df = pd.DataFrame(rows)
    df["holding_cost"] *= hold_mult
    df["lost_margin"] *= lost_mult
    df["total_cost"] = df["holding_cost"] + df["lost_margin"]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--service-level", type=float, default=0.95)
    ap.add_argument("--activate", action="store_true")
    a = ap.parse_args()
    pairs = load_pairs()
    mu, rmse = planning_inputs(pairs)
    rop, S = optimized_policy(pairs, mu, rmse, a.service_level)
    name = f"optimized_sl{int(round(a.service_level * 100))}"
    with get_engine().begin() as c:
        c.execute(text("DELETE FROM reorder_params WHERE policy_name = :p"), {"p": name})
    copy_df(pd.DataFrame({"sku_id": pairs["sku_id"], "warehouse_id": pairs["warehouse_id"],
                          "policy_name": name, "reorder_point": rop, "order_up_to": S}), "reorder_params")
    print(f"Wrote policy '{name}' for {len(pairs):,} pairs.")
    cmp_ = sweep(pairs, mu, rmse, levels=(a.service_level,))
    print(cmp_.round(3).to_string(index=False))
    if a.activate:
        set_meta("active_policy", name)
        print(f"Active policy is now '{name}' - the next daily simulator run will use it.")


if __name__ == "__main__":
    main()
