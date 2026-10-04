"""Supplier-chaos simulator that writes to Postgres.

    python -m src.simulator backfill --days 90   # first run, builds history
    python -m src.simulator daily                # advances the world to today (idempotent)
"""
import argparse
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import text

from . import config
from .db import copy_df, get_engine, get_meta, load_pairs, pair_lookup, set_meta
from .engine import Engine


def load_policy(pairs, policy_name):
    df = pd.read_sql(text("SELECT sku_id, warehouse_id, reorder_point, order_up_to "
                          "FROM reorder_params WHERE policy_name = :p"),
                     get_engine(), params={"p": policy_name})
    if df.empty:
        raise RuntimeError(f"No reorder_params for policy '{policy_name}'. Run src.seed / src.optimizer first.")
    df = df.set_index(["sku_id", "warehouse_id"]).reindex(pair_lookup(pairs))
    return df["reorder_point"].to_numpy(), df["order_up_to"].to_numpy()


def step_to_frames(out, pairs, policy_name, status_cutoff):
    """Turn one engine step into row-frames for each table."""
    d = out["date"]
    n = len(pairs)
    sku, wh = pairs["sku_id"].to_numpy(), pairs["warehouse_id"].to_numpy()
    sales = pd.DataFrame({"sale_date": d, "sku_id": sku, "warehouse_id": wh,
                          "units_demanded": out["demand"], "units_sold": out["sold"]})
    snap = pd.DataFrame({"snapshot_date": d, "sku_id": sku, "warehouse_id": wh,
                         "on_hand": out["on_hand"], "on_order": out["on_order"]})
    m = out["lost"] > 0
    stock = pd.DataFrame({"event_date": d, "sku_id": sku[m], "warehouse_id": wh[m],
                          "lost_units": out["lost"][m]})
    i = out["order_idx"]
    arr = [d + timedelta(days=int(x)) for x in out["order_lt"]]
    exp = [d + timedelta(days=int(x)) for x in out["order_exp_lt"]]
    po = pd.DataFrame({"sku_id": sku[i], "warehouse_id": wh[i], "order_date": d,
                       "expected_arrival": exp, "actual_arrival": arr, "qty": out["order_qty"]})
    po["status"] = np.where(pd.to_datetime(po["actual_arrival"]) <= pd.Timestamp(status_cutoff),
                            "received", "open")
    kpi = pd.DataFrame([{"kpi_date": d, "policy_name": policy_name, **out["kpi"]}])
    return dict(sales_daily=sales, inventory_snapshot=snap, stockout_events=stock,
                purchase_orders=po, daily_kpis=kpi)


def flush(buffers):
    for table, frames in buffers.items():
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)
        copy_df(df, table)
        frames.clear()


def backfill(days):
    with get_engine().connect() as c:
        if c.execute(text("SELECT COUNT(*) FROM sales_daily")).scalar() > 0:
            raise SystemExit("sales_daily already has data. Re-run sql/01_schema.sql and src.seed to start over.")
    pairs = load_pairs()
    policy = get_meta("active_policy", config.BASELINE_POLICY)
    rop, S = load_policy(pairs, policy)
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=days - 1)
    set_meta("sim_origin", start.isoformat())
    eng = Engine(pairs, rop, S, on_hand=S, pipe=np.zeros((len(pairs), config.MAX_LEAD), dtype=np.int64),
                 origin_date=start, seed=config.SEED)
    buffers = {t: [] for t in ["sales_daily", "inventory_snapshot", "stockout_events",
                               "purchase_orders", "daily_kpis"]}
    d = start
    while d <= end:
        out = eng.step(d)
        for t, f in step_to_frames(out, pairs, policy, status_cutoff=end).items():
            buffers[t].append(f)
        if (d - start).days % 10 == 9:
            flush(buffers)
            print(f"  ... simulated through {d}")
        d += timedelta(days=1)
    flush(buffers)
    print(f"Backfill complete: {days} days ({start} -> {end}), policy '{policy}'.")


def load_state(pairs, last_day, next_day):
    """Rebuild on-hand and the supplier pipeline from the database."""
    idx = pair_lookup(pairs)
    snap = pd.read_sql(text("SELECT sku_id, warehouse_id, on_hand FROM inventory_snapshot "
                            "WHERE snapshot_date = :d"), get_engine(), params={"d": last_day})
    on_hand = snap.set_index(["sku_id", "warehouse_id"])["on_hand"].reindex(idx).fillna(0).to_numpy()
    po = pd.read_sql(text("SELECT sku_id, warehouse_id, (actual_arrival - :nd) AS k, SUM(qty) AS qty "
                          "FROM purchase_orders WHERE status = 'open' GROUP BY 1, 2, 3"),
                     get_engine(), params={"nd": next_day})
    pipe = np.zeros((len(pairs), config.MAX_LEAD), dtype=np.int64)
    if not po.empty:
        rows = idx.get_indexer(pd.MultiIndex.from_arrays([po["sku_id"], po["warehouse_id"]]))
        cols = np.clip(po["k"].to_numpy(int), 0, config.MAX_LEAD - 1)
        np.add.at(pipe, (rows, cols), po["qty"].to_numpy(int))
    return on_hand, pipe


def daily():
    last = get_engine().connect().execute(text("SELECT MAX(sale_date) FROM sales_daily")).scalar()
    if last is None:
        raise SystemExit("No history yet. Run: python -m src.simulator backfill")
    nxt = last + timedelta(days=1)
    if nxt > date.today():
        print(f"Already up to date (last simulated day: {last}).")
        return
    pairs = load_pairs()
    policy = get_meta("active_policy", config.BASELINE_POLICY)
    rop, S = load_policy(pairs, policy)
    origin = date.fromisoformat(get_meta("sim_origin"))
    on_hand, pipe = load_state(pairs, last, nxt)
    eng = Engine(pairs, rop, S, on_hand, pipe, origin, seed=config.SEED + (nxt - origin).days)
    d = nxt
    while d <= date.today():
        out = eng.step(d)
        with get_engine().begin() as c:
            c.execute(text("UPDATE purchase_orders SET status = 'received' "
                           "WHERE status = 'open' AND actual_arrival <= :d"), {"d": d})
        for t, f in step_to_frames(out, pairs, policy, status_cutoff=d).items():
            copy_df(f, t)
        k = out["kpi"]
        print(f"{d}: policy={policy} fill_rate={k['fill_rate']:.3f} lost_units={k['lost_units']} "
              f"orders={k['orders_placed']}")
        d += timedelta(days=1)
    with get_engine().begin() as c:   # keep a cloud database small
        c.execute(text("DELETE FROM inventory_snapshot WHERE snapshot_date < :c"),
                  {"c": date.today() - timedelta(days=config.SNAPSHOT_RETENTION_DAYS)})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backfill")
    b.add_argument("--days", type=int, default=config.BACKFILL_DAYS)
    sub.add_parser("daily")
    a = ap.parse_args()
    backfill(a.days) if a.cmd == "backfill" else daily()
