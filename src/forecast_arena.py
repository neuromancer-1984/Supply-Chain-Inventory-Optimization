"""Forecast arena: 4 models compete on rolling-origin backtests; errors are stored in SQL.

    python -m src.forecast_arena backtest   # score every model on every fold -> forecast_accuracy
    python -m src.forecast_arena live       # forecast next 7 days with each category's champion
    python -m src.forecast_arena all

Models: naive, seasonal_naive, ets_lite (SES x weekly index), lightgbm (global, Poisson objective).
All forecasts for target day d use data up to the origin day only (no leakage).
"""
import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd
from sqlalchemy import text

from . import config
from .db import copy_df, get_engine, load_pairs, pair_lookup

H = config.HORIZON
assert 1 <= H <= 7, "horizon must be 1..7 (seasonal lag logic)"


# ------------------------------------------------------------------ data
def load_matrix(pairs):
    df = pd.read_sql("SELECT sale_date, sku_id, warehouse_id, units_demanded FROM sales_daily", get_engine())
    df["sale_date"] = pd.to_datetime(df["sale_date"])
    dates = pd.DatetimeIndex(np.sort(df["sale_date"].unique()))
    rows = pair_lookup(pairs).get_indexer(pd.MultiIndex.from_arrays([df["sku_id"], df["warehouse_id"]]))
    cols = dates.get_indexer(df["sale_date"])
    Y = np.zeros((len(pairs), len(dates)), dtype=np.float32)
    Y[rows, cols] = df["units_demanded"].to_numpy()
    return Y, dates


# ------------------------------------------------------------------ simple models
def naive(Y, O, dow0):
    return np.repeat(Y[:, O:O + 1], H, axis=1)


def seasonal_naive(Y, O, dow0):
    return np.stack([Y[:, O + h - 7] for h in range(1, H + 1)], axis=1)


def ets_lite(Y, O, dow0, alpha=0.2):
    """Simple exponential smoothing on a deseasonalised series x shrunk weekly index."""
    hist = Y[:, :O + 1].astype(np.float64)
    dow_hist = (dow0 + np.arange(O + 1)) % 7
    last, dow_last = hist[:, -28:], dow_hist[-28:]
    overall = last.mean(axis=1, keepdims=True) + 1e-6
    seas = np.stack([last[:, dow_last == d].mean(axis=1) for d in range(7)], axis=1) / overall
    seas = 0.5 * seas + 0.5                                  # shrink noisy low-volume indices
    ds = hist / np.maximum(seas[:, dow_hist], 0.05)
    level = ds[:, :7].mean(axis=1)
    for t in range(7, ds.shape[1]):
        level = alpha * ds[:, t] + (1 - alpha) * level
    dow_future = (dow0 + O + np.arange(1, H + 1)) % 7
    return level[:, None] * seas[:, dow_future]


# ------------------------------------------------------------------ LightGBM
class GlobalLGB:
    def __init__(self, Y, pairs, dow0, seed=config.SEED):
        self.Y, self.dow0 = Y, dow0
        n, T = Y.shape
        C = np.cumsum(np.pad(Y.astype(np.float64), ((0, 0), (1, 0))), axis=1)
        t = np.arange(T)

        def roll(w):
            lo = np.maximum(0, t + 1 - w)
            return ((C[:, t + 1] - C[:, lo]) / (t + 1 - lo)).astype(np.float32)

        self.m7, self.m28 = roll(7), roll(28)
        cats = {c: i for i, c in enumerate(config.CATEGORIES)}
        self.cat = pairs["category"].map(cats).to_numpy(np.float32)
        self.wh = pairs["warehouse_id"].to_numpy(np.float32)
        self.lcost = np.log(pairs["unit_cost"].to_numpy(np.float32))
        self.rng = np.random.default_rng(seed)

    def feats(self, o, h, sel):
        Y = self.Y
        same_dow = np.mean([Y[sel, o + h - 7 * k] for k in (1, 2, 3, 4)], axis=0)
        m7, m28 = self.m7[sel, o], self.m28[sel, o]
        return np.column_stack([
            Y[sel, o], m7, m28, m7 / (m28 + 0.05), Y[sel, o + h - 7], same_dow,
            np.full(len(sel), (self.dow0 + o + h) % 7, np.float32), np.full(len(sel), h, np.float32),
            self.cat[sel], self.wh[sel], self.lcost[sel]]).astype(np.float32)

    def fit_predict(self, O):
        n = self.Y.shape[0]
        sel = self.rng.choice(n, size=min(config.LGB_SERIES_SAMPLE, n), replace=False)
        X, y = [], []
        for o in range(27, O):
            for h in range(1, H + 1):
                if o + h <= O:
                    X.append(self.feats(o, h, sel))
                    y.append(self.Y[sel, o + h])
        X, y = np.vstack(X), np.concatenate(y)
        if len(y) > config.LGB_MAX_ROWS:
            keep = self.rng.choice(len(y), config.LGB_MAX_ROWS, replace=False)
            X, y = X[keep], y[keep]
        model = lgb.LGBMRegressor(objective="poisson", n_estimators=150, learning_rate=0.08,
                                  num_leaves=31, min_child_samples=100, subsample=0.8,
                                  subsample_freq=1, colsample_bytree=0.9, verbose=-1, n_jobs=-1)
        model.fit(X, y)
        allrows = np.arange(n)
        return np.stack([model.predict(self.feats(O, h, allrows)) for h in range(1, H + 1)], axis=1)


def all_forecasts(Y, pairs, dow0, O, lgb_model):
    return {"naive": naive(Y, O, dow0), "seasonal_naive": seasonal_naive(Y, O, dow0),
            "ets_lite": ets_lite(Y, O, dow0), "lightgbm": lgb_model.fit_predict(O)}


# ------------------------------------------------------------------ commands
def backtest():
    pairs = load_pairs()
    Y, dates = load_matrix(pairs)
    T, dow0 = Y.shape[1], dates[0].dayofweek
    origins = list(range(config.FIRST_ORIGIN_IDX, T - H, config.FOLD_STEP))
    if not origins:
        raise SystemExit(f"Need more history: have {T} days, need at least "
                         f"{config.FIRST_ORIGIN_IDX + H + 1}. Run a longer backfill.")
    with get_engine().begin() as c:
        c.execute(text("DELETE FROM forecast_runs WHERE kind = 'backtest'"))
    glgb = GlobalLGB(Y, pairs, dow0)
    for O in origins:
        preds = all_forecasts(Y, pairs, dow0, O, glgb)
        actual = Y[:, O + 1:O + H + 1].astype(np.float64)
        for model, p in preds.items():
            err = p - actual
            with get_engine().begin() as c:
                run_id = c.execute(text(
                    "INSERT INTO forecast_runs (kind, model_name, origin_date, horizon_days) "
                    "VALUES ('backtest', :m, :o, :h) RETURNING run_id"),
                    {"m": model, "o": dates[O].date(), "h": H}).scalar()
            copy_df(pd.DataFrame({
                "run_id": run_id, "sku_id": pairs["sku_id"], "warehouse_id": pairs["warehouse_id"],
                "abs_err_sum": np.abs(err).sum(axis=1), "sq_err_sum": (err ** 2).sum(axis=1),
                "actual_sum": actual.sum(axis=1), "n_obs": H}), "forecast_accuracy")
        wape = {m: np.abs(p - actual).sum() / actual.sum() for m, p in preds.items()}
        print(f"fold origin {dates[O].date()}: " + "  ".join(f"{m}={v:.3f}" for m, v in wape.items()))
    print("Backtest done. See: SELECT * FROM v_model_category_accuracy; SELECT * FROM v_champion_model;")


def live():
    champ = pd.read_sql("SELECT category, model_name FROM v_champion_model", get_engine())
    if champ.empty:
        raise SystemExit("No champion yet. Run: python -m src.forecast_arena backtest")
    pairs = load_pairs()
    Y, dates = load_matrix(pairs)
    O, dow0 = Y.shape[1] - 1, dates[0].dayofweek
    preds = all_forecasts(Y, pairs, dow0, O, GlobalLGB(Y, pairs, dow0))
    chosen = pairs["category"].map(dict(zip(champ["category"], champ["model_name"])))
    out = np.zeros((len(pairs), H))
    for model, p in preds.items():
        m = (chosen == model).to_numpy()
        out[m] = p[m]
    with get_engine().begin() as c:
        c.execute(text("DELETE FROM forecast_runs WHERE kind = 'live'"))
        run_id = c.execute(text(
            "INSERT INTO forecast_runs (kind, model_name, origin_date, horizon_days) "
            "VALUES ('live', 'champion', :o, :h) RETURNING run_id"),
            {"o": dates[O].date(), "h": H}).scalar()
    frames = []
    for h in range(1, H + 1):
        frames.append(pd.DataFrame({
            "run_id": run_id, "model_name": chosen.to_numpy(), "sku_id": pairs["sku_id"],
            "warehouse_id": pairs["warehouse_id"],
            "forecast_date": (dates[O] + pd.Timedelta(days=h)).date(),
            "predicted_units": np.maximum(out[:, h - 1], 0)}))
    copy_df(pd.concat(frames, ignore_index=True), "forecasts")
    print(f"Live forecast written for {dates[O].date()} + {H} days "
          f"({len(pairs):,} pairs, champion model per category).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["backtest", "live", "all"])
    a = ap.parse_args()
    if a.cmd in ("backtest", "all"):
        backtest()
    if a.cmd in ("live", "all"):
        live()
