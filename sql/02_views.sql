-- 02_views.sql : analytics layer. Safe to re-run (CREATE OR REPLACE).
-- Run AFTER 01_schema.sql. Views that depend on forecasts return rows once the
-- forecast arena has run.

-- 1. Forecast accuracy per category and model (all backtest folds pooled)
CREATE OR REPLACE VIEW v_model_category_accuracy AS
SELECT s.category,
       fr.model_name,
       SUM(fa.abs_err_sum) / NULLIF(SUM(fa.actual_sum), 0)            AS wape,
       SUM(fa.abs_err_sum) / NULLIF(SUM(fa.n_obs), 0)                 AS mae,
       SQRT(SUM(fa.sq_err_sum) / NULLIF(SUM(fa.n_obs), 0))            AS rmse,
       COUNT(DISTINCT fr.origin_date)                                 AS folds
FROM forecast_accuracy fa
JOIN forecast_runs fr USING (run_id)
JOIN skus s USING (sku_id)
WHERE fr.kind = 'backtest'
GROUP BY s.category, fr.model_name;

-- 2. Champion model per category (lowest WAPE), via window function
CREATE OR REPLACE VIEW v_champion_model AS
SELECT category, model_name, wape, mae
FROM (
    SELECT *, RANK() OVER (PARTITION BY category ORDER BY wape ASC) AS rk
    FROM v_model_category_accuracy
) t
WHERE rk = 1;

-- 3. Forecast error (RMSE) of each SKU-warehouse pair under its category champion
CREATE OR REPLACE VIEW v_pair_champion_error AS
SELECT fa.sku_id,
       fa.warehouse_id,
       SQRT(SUM(fa.sq_err_sum) / NULLIF(SUM(fa.n_obs), 0)) AS rmse
FROM forecast_accuracy fa
JOIN forecast_runs fr USING (run_id)
JOIN skus s USING (sku_id)
JOIN v_champion_model c ON c.category = s.category AND c.model_name = fr.model_name
WHERE fr.kind = 'backtest'
GROUP BY fa.sku_id, fa.warehouse_id;

-- 4. Expected daily demand per pair from the latest live forecast
CREATE OR REPLACE VIEW v_latest_forecast_mu AS
SELECT sku_id, warehouse_id, AVG(predicted_units) AS mu
FROM forecasts
WHERE run_id IN (SELECT MAX(run_id) FROM forecast_runs WHERE kind = 'live')
GROUP BY sku_id, warehouse_id;

-- 5. ABC classification on last-30-day revenue (cumulative share window)
CREATE OR REPLACE VIEW v_abc_class AS
WITH rev AS (
    SELECT sd.sku_id, SUM(sd.units_sold * s.unit_price) AS revenue
    FROM sales_daily sd
    JOIN skus s USING (sku_id)
    WHERE sd.sale_date > (SELECT MAX(sale_date) FROM sales_daily) - 30
    GROUP BY sd.sku_id
), ranked AS (
    SELECT sku_id, revenue,
           SUM(revenue) OVER (ORDER BY revenue DESC, sku_id) / SUM(revenue) OVER () AS cum_share
    FROM rev
)
SELECT sku_id, revenue, cum_share,
       CASE WHEN cum_share <= 0.80 THEN 'A'
            WHEN cum_share <= 0.95 THEN 'B'
            ELSE 'C' END AS abc_class
FROM ranked;

-- 6. Inventory health: days of cover from the latest snapshot
CREATE OR REPLACE VIEW v_inventory_health AS
WITH last_day AS (SELECT MAX(snapshot_date) AS d FROM inventory_snapshot),
dem AS (
    SELECT sku_id, warehouse_id, AVG(units_demanded)::numeric AS avg_daily
    FROM sales_daily
    WHERE sale_date > (SELECT MAX(sale_date) FROM sales_daily) - 14
    GROUP BY sku_id, warehouse_id
)
SELECT i.sku_id, i.warehouse_id, s.sku_code, s.category,
       i.on_hand, i.on_order,
       ROUND(d.avg_daily, 2)                              AS avg_daily_demand,
       ROUND(i.on_hand / NULLIF(d.avg_daily, 0), 1)       AS days_of_cover,
       CASE WHEN i.on_hand = 0                                 THEN 'STOCKOUT'
            WHEN i.on_hand / NULLIF(d.avg_daily, 0) < 3        THEN 'LOW'
            WHEN i.on_hand / NULLIF(d.avg_daily, 0) > 45       THEN 'OVERSTOCK'
            ELSE 'OK' END                                  AS status
FROM inventory_snapshot i
JOIN last_day l ON i.snapshot_date = l.d
JOIN skus s USING (sku_id)
JOIN dem d USING (sku_id, warehouse_id);

-- 7. Supplier performance on received orders
CREATE OR REPLACE VIEW v_supplier_performance AS
SELECT su.supplier_id, su.name,
       COUNT(*)                                                    AS orders,
       ROUND(AVG(po.actual_arrival - po.expected_arrival), 2)      AS avg_delay_days,
       ROUND(AVG((po.actual_arrival > po.expected_arrival)::int), 3) AS late_rate
FROM purchase_orders po
JOIN skus s USING (sku_id)
JOIN suppliers su USING (supplier_id)
WHERE po.status = 'received'
GROUP BY su.supplier_id, su.name;

-- 8. Policy comparison from daily KPIs
CREATE OR REPLACE VIEW v_policy_comparison AS
SELECT policy_name,
       MIN(kpi_date)                                   AS first_day,
       MAX(kpi_date)                                   AS last_day,
       COUNT(*)                                        AS days,
       ROUND(SUM(units_sold)::numeric / NULLIF(SUM(units_demanded), 0), 4) AS fill_rate,
       ROUND(AVG(inventory_value), 0)                  AS avg_inventory_value,
       ROUND(SUM(holding_cost), 0)                     AS holding_cost,
       ROUND(SUM(lost_margin), 0)                      AS lost_margin,
       ROUND(SUM(holding_cost) + SUM(lost_margin), 0)  AS total_cost
FROM daily_kpis
GROUP BY policy_name;
