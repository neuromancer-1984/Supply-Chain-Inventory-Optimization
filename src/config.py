"""Central settings. Change scale here."""
import numpy as np

# ---- scale ----
N_SKUS = 10_000
N_SUPPLIERS = 40
BACKFILL_DAYS = 90          # history generated on first run (60-120 is sensible locally)
HORIZON = 7                 # forecast horizon in days (max 7)
SNAPSHOT_RETENTION_DAYS = 21  # pruned only by the daily job (keeps cloud DB small)

# ---- simulation ----
DELAY_DAYS = 3              # extra days when a supplier order arrives late
MAX_LEAD = 32               # pipeline length in days
BASELINE_POLICY = "baseline"
SEED = 42

WAREHOUSES = [
    ("WH-MUM", "Mumbai", 1.30),
    ("WH-DEL", "Delhi", 1.10),
    ("WH-BLR", "Bengaluru", 0.90),
]

# category -> (share of SKUs, median unit cost, weekly demand shape Mon..Sun)
CATEGORIES = {
    "Grocery":     (0.25, 120,  [1.00, 0.95, 0.95, 1.00, 1.10, 1.15, 0.85]),
    "Electronics": (0.15, 4500, [0.90, 0.90, 0.95, 1.00, 1.10, 1.25, 0.90]),
    "Apparel":     (0.20, 900,  [0.85, 0.85, 0.90, 0.95, 1.10, 1.35, 1.00]),
    "Home":        (0.15, 1500, [0.90, 0.90, 0.95, 1.00, 1.05, 1.20, 1.00]),
    "Health":      (0.15, 350,  [1.10, 1.05, 1.00, 1.00, 1.00, 0.90, 0.95]),
    "Toys":        (0.10, 600,  [0.80, 0.80, 0.85, 0.90, 1.05, 1.50, 1.10]),
}
# normalise shapes so each averages exactly 1.0
DOW_FACTORS = {k: np.array(v[2]) / np.mean(v[2]) for k, v in CATEGORIES.items()}

# ---- forecast arena ----
MODELS = ["naive", "seasonal_naive", "ets_lite", "lightgbm"]
FIRST_ORIGIN_IDX = 41       # earliest backtest origin (needs 42 days of history)
FOLD_STEP = 7
LGB_SERIES_SAMPLE = 5000
LGB_MAX_ROWS = 600_000
