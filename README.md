# Supply Chain Inventory Optimization & Demand Forecasting

Full technical write-up: [docs/TECHNICAL.md](docs/TECHNICAL.md)

A simulated supply chain (10,000 SKUs x 3 warehouses = 30,000 stocking points) that runs on Postgres,
forecasts demand with competing models, and optimizes reorder policies.

**Four layers**

1. **Simulator** (`src/engine.py`, `src/simulator.py`): daily sales, purchase orders, random supplier
   lead times and delays, stockouts. Advances one day per run, so the data keeps changing.
2. **Forecast arena** (`src/forecast_arena.py`): naive, seasonal naive, ETS-lite and a global LightGBM
   (Poisson objective) compete on rolling-origin 7-day-ahead backtests. Errors are stored in SQL; a
   window-function view picks the champion model per category.
3. **Optimizer** (`src/policy.py`, `src/optimizer.py`): safety stock from forecast error *and*
   lead-time variability, `SS = z * sqrt(L * sigma_d^2 + mu^2 * sigma_L^2)`. A cost sweep finds the
   service level that minimises holding cost + lost margin.
4. **Dashboard** (`app/dashboard.py`): KPIs, inventory health, forecast arena, what-if simulator,
   Excel export (Excel is used for report downloads only).

SQL-heavy parts live in `sql/02_views.sql` (CTEs, window functions, ABC classification,
champion selection, days-of-cover, supplier performance).

## Setup (Windows, pgAdmin 4)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env      # then edit DATABASE_URL
```

1. In pgAdmin create an empty database `supply_chain`.
2. In pgAdmin Query Tool run `sql/01_schema.sql`, then `sql/02_views.sql`.
3. Build the world:

```powershell
python -m src.seed                         # warehouses, suppliers, 10K SKUs, baseline policy
python -m src.simulator backfill --days 90 # ~20 s, ~2.7M sales rows
python -m src.forecast_arena all           # ~1-2 min: backtests + live forecast
python -m src.optimizer --service-level 0.95 --activate
python -m src.simulator daily              # advance to today under the new policy
streamlit run app/dashboard.py
```

After `--activate`, each daily run uses the optimized policy, so the dashboard's fill-rate chart shows
the baseline period followed by the improvement.

## Useful queries to try in pgAdmin

```sql
SELECT * FROM v_model_category_accuracy ORDER BY category, wape;
SELECT * FROM v_champion_model;
SELECT * FROM v_policy_comparison;
SELECT status, COUNT(*) FROM v_inventory_health GROUP BY status;
SELECT * FROM v_supplier_performance ORDER BY late_rate DESC LIMIT 10;
```

## Deploying (same pattern as the MCX dashboard)

- Neon free tier is 0.5 GB. 90 days of history is about 2.7M sales rows, which is too big. Either lower
  `N_SKUS` / `BACKFILL_DAYS` in `src/config.py` (e.g. 3,000 SKUs, 60 days) or keep the full dataset local.
- Add `DATABASE_URL` as a GitHub Actions secret (daily job in `.github/workflows/daily_sim.yml`) and as a
  Streamlit Cloud secret.

## Notes on the numbers

- Demand is simulated with weekly seasonality, trends, promo spikes and overdispersed Poisson noise, so
  per-SKU demand is low-volume and noisy. Accuracy is reported as **WAPE** (sum of absolute errors / sum
  of actual demand), which behaves better than MAPE on small counts.
- Forecasts are scored against true simulated demand (`units_demanded`). With real data you would
  correct sales for stockout censoring first.
- Quote the figures your own run produces. Do not copy numbers from this README into your resume.
