"""Vectorised inventory simulation engine (one step = one day for ALL pairs).

Used by the DB-backed daily simulator AND the in-memory what-if optimizer,
so both always follow identical rules.

Pipeline convention: at the start of a day, pipe[:, k] = units arriving k days from today.
"""
from datetime import date

import numpy as np

from . import config
from .db import dow_matrix


class Engine:
    def __init__(self, pairs, rop, order_up_to, on_hand, pipe, origin_date: date,
                 seed=0, lt_mult=1.0):
        self.p = pairs
        self.n = len(pairs)
        self.base_mu = pairs["base_mu"].to_numpy(float)
        self.trend = pairs["trend_pct"].to_numpy(float) / 100.0 / 30.0
        self.dow = dow_matrix(pairs)
        self.avg_lt = pairs["avg_lt"].to_numpy(float) * lt_mult
        self.lt_std = pairs["lt_std"].to_numpy(float) * lt_mult
        self.delay_p = pairs["delay_prob"].to_numpy(float)
        self.cost = pairs["unit_cost"].to_numpy(float)
        self.price = pairs["unit_price"].to_numpy(float)
        self.hold_daily = pairs["holding_cost_pct"].to_numpy(float) / 365.0
        self.rop = np.asarray(rop, dtype=np.int64)
        self.S = np.asarray(order_up_to, dtype=np.int64)
        self.on_hand = np.asarray(on_hand, dtype=np.int64).copy()
        self.pipe = np.asarray(pipe, dtype=np.int64).copy()
        self.origin = origin_date
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    def expected_demand(self, d: date):
        t = (d - self.origin).days
        mult = np.clip(1.0 + self.trend * t, 0.2, None)
        return self.base_mu * self.dow[:, d.weekday()] * mult

    def step(self, d: date) -> dict:
        rng = self.rng
        # 1. receive deliveries scheduled for today
        self.on_hand += self.pipe[:, 0]
        self.pipe[:, 0] = 0

        # 2. demand: gamma-Poisson (overdispersed) with occasional promo spikes
        mu = self.expected_demand(d)
        promo = np.where(rng.random(self.n) < 0.005, 3.0, 1.0)
        lam = mu * promo * rng.gamma(6.0, 1.0 / 6.0, self.n)
        demand = rng.poisson(lam)
        sold = np.minimum(demand, self.on_hand)
        lost = demand - sold
        self.on_hand -= sold

        # 3. (s, S) ordering on inventory position
        on_order = self.pipe.sum(axis=1)
        position = self.on_hand + on_order
        qty = np.where(position <= self.rop, self.S - position, 0)
        idx = np.nonzero(qty > 0)[0]
        exp_lt = np.clip(np.rint(self.avg_lt[idx]), 1, config.MAX_LEAD - 1).astype(int)
        base = rng.normal(self.avg_lt[idx], np.maximum(self.lt_std[idx], 1e-6))
        late = rng.random(len(idx)) < self.delay_p[idx]
        lt = np.clip(np.rint(base + late * config.DELAY_DAYS), 1, config.MAX_LEAD - 1).astype(int)
        self.pipe[idx, lt] += qty[idx]

        out = dict(
            date=d, demand=demand, sold=sold, lost=lost,
            order_idx=idx, order_qty=qty[idx], order_lt=lt, order_exp_lt=exp_lt,
            on_hand=self.on_hand.copy(), on_order=self.pipe.sum(axis=1),
        )
        out["kpi"] = self._kpis(out)

        # 4. advance the calendar
        self.pipe[:, :-1] = self.pipe[:, 1:]
        self.pipe[:, -1] = 0
        return out

    def _kpis(self, o) -> dict:
        dem, sold, lost = o["demand"].sum(), o["sold"].sum(), o["lost"].sum()
        return dict(
            units_demanded=int(dem), units_sold=int(sold), lost_units=int(lost),
            fill_rate=float(sold / dem) if dem else 1.0,
            stockout_pairs=int((o["lost"] > 0).sum()),
            orders_placed=int(len(o["order_idx"])),
            inventory_value=float((o["on_hand"] * self.cost).sum()),
            holding_cost=float((o["on_hand"] * self.cost * self.hold_daily).sum()),
            lost_margin=float((o["lost"] * (self.price - self.cost)).sum()),
        )
