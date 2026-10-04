"""Database helpers: engine, fast COPY loader, sim_meta, pair table."""
import io
import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from . import config

load_dotenv()

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        url = os.environ.get("DATABASE_URL")
        if not url:
            raise RuntimeError("DATABASE_URL is not set. Copy .env.example to .env and edit it.")
        _engine = create_engine(url, pool_pre_ping=True)
    return _engine


def copy_df(df: pd.DataFrame, table: str, columns=None):
    """Bulk-load a DataFrame with COPY (orders of magnitude faster than INSERT)."""
    if df.empty:
        return
    columns = columns or list(df.columns)
    buf = io.StringIO()
    df[columns].to_csv(buf, index=False, header=False)
    buf.seek(0)
    raw = get_engine().raw_connection()
    try:
        with raw.cursor() as cur:
            cur.copy_expert(f"COPY {table} ({','.join(columns)}) FROM STDIN WITH CSV", buf)
        raw.commit()
    finally:
        raw.close()


def get_meta(key, default=None):
    with get_engine().connect() as c:
        row = c.execute(text("SELECT value FROM sim_meta WHERE key = :k"), {"k": key}).fetchone()
    return row[0] if row else default


def set_meta(key, value):
    with get_engine().begin() as c:
        c.execute(text(
            "INSERT INTO sim_meta (key, value) VALUES (:k, :v) "
            "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value"), {"k": key, "v": str(value)})


PAIRS_SQL = """
SELECT s.sku_id, w.warehouse_id, s.category, s.unit_cost::float AS unit_cost,
       s.unit_price::float AS unit_price, s.holding_cost_pct::float AS holding_cost_pct,
       (s.base_daily_demand * w.demand_factor)::float AS base_mu,
       s.trend_pct_per_month::float AS trend_pct,
       s.supplier_id,
       su.avg_lead_time_days::float AS avg_lt,
       su.lead_time_std_days::float AS lt_std,
       su.delay_prob::float AS delay_prob
FROM skus s
CROSS JOIN warehouses w
JOIN suppliers su USING (supplier_id)
ORDER BY s.sku_id, w.warehouse_id
"""


def load_pairs() -> pd.DataFrame:
    """One row per SKU x warehouse. Row position == pair index used everywhere."""
    df = pd.read_sql(PAIRS_SQL, get_engine())
    df["dow_matrix_idx"] = df["category"].map({c: i for i, c in enumerate(config.CATEGORIES)})
    return df


def dow_matrix(pairs: pd.DataFrame) -> np.ndarray:
    cats = list(config.CATEGORIES)
    table = np.stack([config.DOW_FACTORS[c] for c in cats])      # (n_cat, 7)
    return table[pairs["dow_matrix_idx"].to_numpy()]              # (n_pairs, 7)


def pair_lookup(pairs: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_arrays([pairs["sku_id"], pairs["warehouse_id"]])
