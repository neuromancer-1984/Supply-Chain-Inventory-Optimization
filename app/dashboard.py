"""Streamlit dashboard:  streamlit run app/dashboard.py"""
import io
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import streamlit as st

# Streamlit Cloud: put DATABASE_URL in the app's secrets
try:
    if "DATABASE_URL" in st.secrets:
        os.environ["DATABASE_URL"] = st.secrets["DATABASE_URL"]
except Exception:
    pass

from src.db import get_engine, load_pairs  # noqa: E402
from src.optimizer import planning_inputs, sweep  # noqa: E402

st.set_page_config(page_title="Supply Chain Inventory Optimization", layout="wide")


@st.cache_data(ttl=300, show_spinner=False)
def q(sql: str) -> pd.DataFrame:
    return pd.read_sql(sql, get_engine())


@st.cache_resource(show_spinner="Loading planning data...")
def planning_data():
    pairs = load_pairs()
    mu, rmse = planning_inputs(pairs)
    return pairs, mu, rmse


@st.cache_data(show_spinner="Simulating policies...")
def run_sweep(lt_mult, hold_mult, lost_mult, n):
    pairs, mu, rmse = planning_data()
    return sweep(pairs, mu, rmse, lt_mult=lt_mult, hold_mult=hold_mult, lost_mult=lost_mult, n=n)


st.title("Supply Chain Inventory Optimization & Demand Forecasting")
st.caption("10K SKUs x 3 warehouses | live simulated supply chain in Postgres | "
           "forecast arena + reorder-policy optimizer")

kpis = q("SELECT * FROM daily_kpis ORDER BY kpi_date")
if kpis.empty:
    st.error("No data yet. Run the seed and backfill steps from the README.")
    st.stop()

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "Inventory health", "Forecast arena", "What-if optimizer", "Reports"])

# ------------------------------------------------------------------ Overview
with tab1:
    last = kpis.iloc[-1]
    last30 = kpis.tail(30)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fill rate (latest day)", f"{last['fill_rate']:.1%}")
    c2.metric("Fill rate (30d)", f"{last30['units_sold'].sum() / last30['units_demanded'].sum():.1%}")
    c3.metric("Lost margin (30d)", f"Rs {last30['lost_margin'].sum() / 1e6:,.1f}M")
    c4.metric("Inventory value", f"Rs {last['inventory_value'] / 1e6:,.0f}M")
    st.caption(f"Data through {last['kpi_date']} | active policy: {last['policy_name']}")

    fig = px.line(kpis, x="kpi_date", y="fill_rate", color="policy_name", title="Daily fill rate by policy")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)
    left, right = st.columns(2)
    left.plotly_chart(px.line(kpis, x="kpi_date", y="inventory_value", color="policy_name",
                              title="Inventory value (Rs)"), use_container_width=True)
    right.plotly_chart(px.bar(kpis, x="kpi_date", y="lost_margin", color="policy_name",
                              title="Margin lost to stockouts (Rs / day)"), use_container_width=True)
    st.subheader("Policy comparison")
    st.dataframe(q("SELECT * FROM v_policy_comparison ORDER BY first_day"), use_container_width=True)

# ------------------------------------------------------------------ Inventory health
with tab2:
    health = q("SELECT status, COUNT(*) AS pairs FROM v_inventory_health GROUP BY status")
    if health.empty:
        st.info("No inventory snapshot found.")
    else:
        a, b = st.columns([1, 2])
        a.plotly_chart(px.bar(health, x="status", y="pairs", color="status", title="SKU-warehouse status"),
                       use_container_width=True)
        abc = q("SELECT abc_class, COUNT(*) AS skus, ROUND(SUM(revenue)) AS revenue "
                "FROM v_abc_class GROUP BY abc_class ORDER BY abc_class")
        b.plotly_chart(px.pie(abc, names="abc_class", values="revenue", title="Revenue share by ABC class"),
                       use_container_width=True)
        risk = q("""
            SELECT h.sku_code, w.city AS warehouse, h.category, a.abc_class, h.on_hand, h.on_order,
                   h.avg_daily_demand, h.days_of_cover, h.status
            FROM v_inventory_health h
            JOIN warehouses w USING (warehouse_id)
            LEFT JOIN v_abc_class a USING (sku_id)
            WHERE h.status IN ('STOCKOUT', 'LOW')
            ORDER BY a.abc_class NULLS LAST, h.days_of_cover NULLS FIRST
            LIMIT 3000""")
        f1, f2 = st.columns(2)
        cat = f1.multiselect("Category", sorted(risk["category"].unique()))
        cls = f2.multiselect("ABC class", ["A", "B", "C"], default=["A"])
        view = risk
        if cat:
            view = view[view["category"].isin(cat)]
        if cls:
            view = view[view["abc_class"].isin(cls)]
        st.subheader("At-risk items (stockout or < 3 days of cover)")
        st.dataframe(view.head(500), use_container_width=True)

# ------------------------------------------------------------------ Forecast arena
with tab3:
    acc = q("SELECT * FROM v_model_category_accuracy")
    if acc.empty:
        st.info("Run `python -m src.forecast_arena all` first.")
    else:
        pooled = q("""SELECT fr.model_name, SUM(fa.abs_err_sum) / SUM(fa.actual_sum) AS wape
                      FROM forecast_accuracy fa JOIN forecast_runs fr USING (run_id)
                      WHERE fr.kind = 'backtest' GROUP BY fr.model_name ORDER BY wape DESC""")
        naive_w = float(pooled.loc[pooled.model_name == "naive", "wape"].iloc[0])
        best = pooled.iloc[-1]
        m1, m2, m3 = st.columns(3)
        m1.metric("Naive WAPE", f"{naive_w:.1%}")
        m2.metric(f"Best model ({best['model_name']}) WAPE", f"{best['wape']:.1%}")
        m3.metric("Error reduction vs naive", f"{1 - best['wape'] / naive_w:.0%}")
        st.caption("WAPE = sum of absolute errors / sum of actual demand, 7-day-ahead forecasts at "
                   "SKU x warehouse x day level, pooled over rolling backtest folds.")
        st.plotly_chart(px.bar(acc, x="category", y="wape", color="model_name", barmode="group",
                               title="Backtest WAPE by category and model"), use_container_width=True)
        st.subheader("Champion model per category")
        st.dataframe(q("SELECT * FROM v_champion_model ORDER BY category"), use_container_width=True)

# ------------------------------------------------------------------ What-if
with tab4:
    st.write("Simulate the baseline rule against forecast-driven policies at several service levels, "
             "then see which level minimises holding cost + lost margin.")
    c1, c2, c3, c4 = st.columns(4)
    lt_mult = c1.slider("Supplier lead-time change", 0.5, 2.0, 1.0, 0.1)
    hold_mult = c2.slider("Holding-cost multiplier", 1.0, 30.0, 1.0, 1.0,
                          help="Capital cost, obsolescence, shelf life. High values favour leaner stock.")
    lost_mult = c3.slider("Stockout-penalty multiplier", 0.25, 3.0, 1.0, 0.25)
    n = c4.selectbox("Pairs simulated", [2000, 3000, 5000], index=1)
    if st.button("Run simulation", type="primary"):
        try:
            res = run_sweep(lt_mult, hold_mult, lost_mult, n)
            opt = res[res.policy != "baseline"]
            best = opt.loc[opt.total_cost.idxmin()]
            base = res[res.policy == "baseline"].iloc[0]
            k1, k2, k3 = st.columns(3)
            k1.metric("Best service level", f"{best['service_level']:.1%}")
            k2.metric("Fill rate vs baseline", f"{best['fill_rate']:.1%}",
                      f"{(best['fill_rate'] - base['fill_rate']) * 100:+.1f} pts")
            k3.metric("Total cost vs baseline", f"{best['total_cost'] / 1e6:,.1f}M",
                      f"{(best['total_cost'] / base['total_cost'] - 1):+.0%}", delta_color="inverse")
            long = opt.melt(id_vars="service_level", value_vars=["holding_cost", "lost_margin", "total_cost"],
                            var_name="component", value_name="cost")
            st.plotly_chart(px.line(long, x="service_level", y="cost", color="component", markers=True,
                                    title="Cost vs target service level (sample, 46 simulated days)"),
                            use_container_width=True)
            st.dataframe(res.round(3), use_container_width=True)
        except Exception as e:
            st.error(f"Simulation failed: {e}. Did you run the forecast arena?")
    else:
        st.info("Set the sliders and press **Run simulation**.")

# ------------------------------------------------------------------ Reports
with tab5:
    st.write("Excel is used only for exports, the analysis lives in SQL.")

    def build_report() -> bytes:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as xw:
            q("""SELECT h.sku_code, w.city AS warehouse, h.category, a.abc_class, h.on_hand, h.on_order,
                        h.avg_daily_demand, h.days_of_cover, h.status
                 FROM v_inventory_health h JOIN warehouses w USING (warehouse_id)
                 LEFT JOIN v_abc_class a USING (sku_id)
                 WHERE h.status IN ('STOCKOUT','LOW')
                 ORDER BY a.abc_class NULLS LAST, h.days_of_cover NULLS FIRST LIMIT 5000"""
              ).to_excel(xw, sheet_name="Reorder now", index=False)
            q("SELECT * FROM v_policy_comparison").to_excel(xw, sheet_name="Policy comparison", index=False)
            q("SELECT * FROM v_model_category_accuracy").to_excel(xw, sheet_name="Model accuracy", index=False)
            q("SELECT * FROM v_supplier_performance ORDER BY late_rate DESC"
              ).to_excel(xw, sheet_name="Suppliers", index=False)
        return buf.getvalue()

    st.download_button("Download Excel report", data=build_report(),
                       file_name="inventory_report.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
