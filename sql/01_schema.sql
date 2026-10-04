-- 01_schema.sql : run once in pgAdmin (Query Tool) on an empty database.
-- WARNING: drops and recreates every table.
DROP TABLE IF EXISTS forecasts, forecast_accuracy, forecast_runs, daily_kpis, stockout_events,
    purchase_orders, inventory_snapshot, sales_daily, reorder_params, skus, suppliers,
    warehouses, sim_meta CASCADE;

CREATE TABLE sim_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE warehouses (
    warehouse_id  SERIAL PRIMARY KEY,
    name          TEXT NOT NULL,
    city          TEXT NOT NULL,
    demand_factor NUMERIC(4,2) NOT NULL
);

CREATE TABLE suppliers (
    supplier_id         SERIAL PRIMARY KEY,
    name                TEXT NOT NULL,
    avg_lead_time_days  NUMERIC(5,2) NOT NULL,
    lead_time_std_days  NUMERIC(5,2) NOT NULL,
    delay_prob          NUMERIC(4,3) NOT NULL      -- chance an order arrives late
);

CREATE TABLE skus (
    sku_id               SERIAL PRIMARY KEY,
    sku_code             TEXT UNIQUE NOT NULL,
    category             TEXT NOT NULL,
    unit_cost            NUMERIC(10,2) NOT NULL,
    unit_price           NUMERIC(10,2) NOT NULL,
    holding_cost_pct     NUMERIC(4,3) NOT NULL,    -- annual, fraction of unit cost
    supplier_id          INT NOT NULL REFERENCES suppliers(supplier_id),
    base_daily_demand    NUMERIC(8,3) NOT NULL,    -- simulator ground truth
    trend_pct_per_month  NUMERIC(5,2) NOT NULL
);

CREATE TABLE reorder_params (
    sku_id         INT NOT NULL REFERENCES skus(sku_id),
    warehouse_id   INT NOT NULL REFERENCES warehouses(warehouse_id),
    policy_name    TEXT NOT NULL,
    reorder_point  INT NOT NULL,
    order_up_to    INT NOT NULL,
    PRIMARY KEY (sku_id, warehouse_id, policy_name)
);

CREATE TABLE sales_daily (
    sale_date       DATE NOT NULL,
    sku_id          INT NOT NULL,
    warehouse_id    INT NOT NULL,
    units_demanded  INT NOT NULL,
    units_sold      INT NOT NULL,
    PRIMARY KEY (sale_date, sku_id, warehouse_id)
);

CREATE TABLE inventory_snapshot (
    snapshot_date  DATE NOT NULL,
    sku_id         INT NOT NULL,
    warehouse_id   INT NOT NULL,
    on_hand        INT NOT NULL,
    on_order       INT NOT NULL,
    PRIMARY KEY (snapshot_date, sku_id, warehouse_id)
);

CREATE TABLE purchase_orders (
    po_id             BIGSERIAL PRIMARY KEY,
    sku_id            INT NOT NULL,
    warehouse_id      INT NOT NULL,
    order_date        DATE NOT NULL,
    expected_arrival  DATE NOT NULL,
    actual_arrival    DATE NOT NULL,
    qty               INT NOT NULL,
    status            TEXT NOT NULL CHECK (status IN ('open','received'))
);
CREATE INDEX idx_po_status ON purchase_orders(status, actual_arrival);

CREATE TABLE stockout_events (
    event_date    DATE NOT NULL,
    sku_id        INT NOT NULL,
    warehouse_id  INT NOT NULL,
    lost_units    INT NOT NULL,
    PRIMARY KEY (event_date, sku_id, warehouse_id)
);

CREATE TABLE daily_kpis (
    kpi_date         DATE PRIMARY KEY,
    policy_name      TEXT NOT NULL,
    units_demanded   BIGINT NOT NULL,
    units_sold       BIGINT NOT NULL,
    lost_units       BIGINT NOT NULL,
    fill_rate        NUMERIC(6,4) NOT NULL,
    stockout_pairs   INT NOT NULL,
    orders_placed    INT NOT NULL,
    inventory_value  NUMERIC(14,2) NOT NULL,
    holding_cost     NUMERIC(14,2) NOT NULL,
    lost_margin      NUMERIC(14,2) NOT NULL
);

CREATE TABLE forecast_runs (
    run_id         SERIAL PRIMARY KEY,
    kind           TEXT NOT NULL CHECK (kind IN ('backtest','live')),
    model_name     TEXT NOT NULL,
    origin_date    DATE NOT NULL,              -- last day of history used
    horizon_days   INT NOT NULL,
    created_at     TIMESTAMP NOT NULL DEFAULT now()
);

CREATE TABLE forecast_accuracy (
    run_id        INT NOT NULL REFERENCES forecast_runs(run_id) ON DELETE CASCADE,
    sku_id        INT NOT NULL,
    warehouse_id  INT NOT NULL,
    abs_err_sum   REAL NOT NULL,
    sq_err_sum    REAL NOT NULL,
    actual_sum    REAL NOT NULL,
    n_obs         SMALLINT NOT NULL,
    PRIMARY KEY (run_id, sku_id, warehouse_id)
);

CREATE TABLE forecasts (
    run_id           INT NOT NULL REFERENCES forecast_runs(run_id) ON DELETE CASCADE,
    model_name       TEXT NOT NULL,
    sku_id           INT NOT NULL,
    warehouse_id     INT NOT NULL,
    forecast_date    DATE NOT NULL,
    predicted_units  REAL NOT NULL,
    PRIMARY KEY (run_id, sku_id, warehouse_id, forecast_date)
);
