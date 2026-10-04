"""Reorder-policy maths: baseline rule vs forecast-driven safety-stock policy."""
import numpy as np
from scipy.stats import norm

from . import config


def baseline_policy(pairs, review_days=7, lt_mult=1.0):
    """Naive planner: cover average demand over lead time, no safety stock."""
    mu = pairs["base_mu"].to_numpy(float)
    L = pairs["avg_lt"].to_numpy(float) * lt_mult
    rop = np.ceil(mu * L)
    S = rop + np.ceil(mu * review_days)
    return rop.astype(int), np.maximum(S, rop + 1).astype(int)


def optimized_policy(pairs, mu, rmse, service_level=0.95, lt_mult=1.0, review_days=7):
    """Safety stock from forecast error AND lead-time variability.

    SS  = z * sqrt(L * sigma_d^2 + mu^2 * sigma_L^2)
    ROP = mu * L + SS          S = ROP + mu * review_days
    """
    mu = np.asarray(mu, float)
    rmse = np.asarray(rmse, float)
    p = pairs["delay_prob"].to_numpy(float)
    L = pairs["avg_lt"].to_numpy(float) * lt_mult + p * config.DELAY_DAYS
    var_L = (pairs["lt_std"].to_numpy(float) * lt_mult) ** 2 + p * (1 - p) * config.DELAY_DAYS ** 2
    z = norm.ppf(service_level)
    ss = z * np.sqrt(L * rmse ** 2 + mu ** 2 * var_L)
    rop = np.ceil(mu * L + ss)
    S = rop + np.ceil(mu * review_days)
    return rop.astype(int), np.maximum(S, rop + 1).astype(int)
