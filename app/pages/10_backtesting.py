"""
QuantEdge — Page 10: Backtesting Studio & Historical Simulation
Simulates systematic portfolio strategies with execution friction, slippage,
transaction costs, and zero-lookahead temporal delays.
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.components.ui import THEME, apply_terminal_theme, create_plotly_figure, render_metric_card
from src.backtesting.engine import BacktestEngine
from src.backtesting.strategies import StrategyLibrary
from src.data.ingestion import DEMO_UNIVERSE, MarketDataIngestion
from src.regimes.regime_detector import MarketRegimeDetector

st.set_page_config(page_title="Backtesting Studio | QuantEdge", page_icon="⚡", layout="wide")
apply_terminal_theme()

if "ingestion" not in st.session_state:
    st.session_state.ingestion = MarketDataIngestion()
ingestion: MarketDataIngestion = st.session_state.ingestion

st.markdown("## ⚡ Backtesting Studio & Historical Simulation")
st.caption("Realistic simulation environment with transaction costs, slippage, and causality safeguards.")

# Sidebar Configuration
with st.sidebar:
    st.markdown("### ⚙️ Simulation Settings")
    init_capital = st.number_input("Initial Capital ($)", min_value=10_000, max_value=10_000_000, value=100_000, step=10_000)
    tx_cost_bps = st.slider("Transaction Fee (bps per trade)", min_value=0.0, max_value=50.0, value=10.0, step=2.5)
    slippage_bps = st.slider("Execution Slippage (bps)", min_value=0.0, max_value=30.0, value=5.0, step=2.5)
    exec_lag = st.selectbox("Execution Delay (Days)", options=[1, 2], index=0, help="Signal at close of day t executes on day t+lag. Enforces causality.")

# Strategy & Universe Controls
c1, c2 = st.columns([2, 1])
with c1:
    default_assets = ["SPY", "QQQ", "AAPL", "NVDA", "GLD", "TLT"]
    selected_assets = st.multiselect("Asset Universe", options=DEMO_UNIVERSE["US"] + DEMO_UNIVERSE["INDIA"], default=default_assets)
with c2:
    strat_choice = st.selectbox(
        "Strategy Framework",
        options=[
            "Periodic Equal-Weighted Rebalance (1/N)",
            "Moving Average Crossover (Trend Following)",
            "Cross-Sectional Momentum (Top K Performers)",
            "Mean Reversion (Oversold Dip-Buying)",
            "Dynamic Risk Parity (Rolling Covariance)",
            "Regime-Aware Adaptive Allocation",
            "Buy & Hold (Static Baseline)",
        ],
        index=0,
    )

if len(selected_assets) < 2:
    st.warning("Please select at least 2 assets to run backtesting.")
    st.stop()

# Retrieve Prices
with st.spinner("Synchronizing historical price matrix..."):
    results = {t: ingestion.fetch_market_data(t) for t in selected_assets}
    prices_df = ingestion.get_aligned_close_matrix(results)
    spy_res = ingestion.fetch_market_data("SPY")
    bench_series = spy_res.df["Close"] if not spy_res.df.empty else None

# Strategy Weight Generation
if strat_choice == "Buy & Hold (Static Baseline)":
    weights_df = StrategyLibrary.buy_and_hold(prices_df)
elif strat_choice == "Periodic Equal-Weighted Rebalance (1/N)":
    weights_df = StrategyLibrary.periodic_equal_weight(prices_df, freq="M")
elif strat_choice == "Moving Average Crossover (Trend Following)":
    weights_df = StrategyLibrary.moving_average_crossover(prices_df, fast_window=20, slow_window=50)
elif strat_choice == "Cross-Sectional Momentum (Top K Performers)":
    weights_df = StrategyLibrary.cross_sectional_momentum(prices_df, lookback_days=63, top_k=2)
elif strat_choice == "Mean Reversion (Oversold Dip-Buying)":
    weights_df = StrategyLibrary.mean_reversion(prices_df, window=20, z_threshold=-1.2)
elif strat_choice == "Dynamic Risk Parity (Rolling Covariance)":
    weights_df = StrategyLibrary.dynamic_risk_parity(prices_df, lookback_days=126, rebalance_freq="M")
elif strat_choice == "Regime-Aware Adaptive Allocation":
    detector = MarketRegimeDetector()
    spy_ret = spy_res.df["Close"].pct_change().dropna()
    reg_res = detector.detect_regimes_hmm(spy_ret, n_regimes=3)
    weights_df = StrategyLibrary.regime_aware_allocation(prices_df, reg_res.regime_labels, defensive_asset="TLT" if "TLT" in prices_df.columns else None)

# Run Backtest
engine = BacktestEngine(
    initial_capital=init_capital,
    transaction_cost_bps=tx_cost_bps,
    slippage_bps=slippage_bps,
    execution_lag=exec_lag,
)

bt_res = engine.run_backtest(prices_df, weights_df, strategy_name=strat_choice, benchmark_series=bench_series)

# Top Performance Metrics
m1, m2, m3, m4, m5, m6 = st.columns(6)
with m1:
    render_metric_card("CAGR", f"{bt_res.cagr:+.2f}%", is_positive=bt_res.cagr >= 0)
with m2:
    render_metric_card("Total Return", f"{bt_res.total_return:+.1f}%", is_positive=bt_res.total_return >= 0)
with m3:
    render_metric_card("Sharpe Ratio", f"{bt_res.sharpe_ratio:.2f}", is_positive=bt_res.sharpe_ratio >= 1.0)
with m4:
    render_metric_card("Sortino Ratio", f"{bt_res.sortino_ratio:.2f}", is_positive=bt_res.sortino_ratio >= 1.0)
with m5:
    render_metric_card("Max Drawdown", f"{bt_res.max_drawdown:.2f}%", is_positive=False)
with m6:
    render_metric_card("Calmar Ratio", f"{bt_res.calmar_ratio:.2f}")

# Main Tabs: Equity, Underwater Drawdown, Monthly Returns, Positions
tab_equity, tab_dd, tab_monthly, tab_pos = st.tabs([
    "📈 Equity Curve vs Benchmark",
    "🌊 Underwater Drawdown",
    "📅 Monthly Returns Heatmap",
    "💼 Position History & Friction",
])

with tab_equity:
    st.markdown("### Cumulative Portfolio Equity Curve ($)")
    fig_eq = create_plotly_figure()
    fig_eq.add_trace(go.Scatter(x=bt_res.equity_curve.index, y=bt_res.equity_curve, name=strat_choice, line=dict(color=THEME["accent_teal"], width=2.5)))
    if bt_res.benchmark_equity_curve is not None:
        fig_eq.add_trace(go.Scatter(x=bt_res.benchmark_equity_curve.index, y=bt_res.benchmark_equity_curve, name="SPY Benchmark", line=dict(color="#718096", width=1.5, dash="dash")))
    fig_eq.update_layout(yaxis_title="Portfolio Value ($)", height=450)
    st.plotly_chart(fig_eq, use_container_width=True)

    if bt_res.alpha is not None and bt_res.beta is not None:
        bc1, bc2, bc3 = st.columns(3)
        with bc1:
            render_metric_card("Jensen's Alpha vs SPY", f"{bt_res.alpha:+.2f}%", is_positive=bt_res.alpha > 0)
        with bc2:
            render_metric_card("Market Beta", f"{bt_res.beta:.2f}")
        with bc3:
            render_metric_card("Information Ratio", f"{bt_res.information_ratio:.2f}" if bt_res.information_ratio is not None else "N/A")

with tab_dd:
    st.markdown("### Historical Drawdown Profile (%)")
    fig_bt_dd = create_plotly_figure()
    fig_bt_dd.add_trace(go.Scatter(
        x=bt_res.drawdown_curve.index,
        y=bt_res.drawdown_curve,
        mode="lines",
        name="Strategy Drawdown",
        fill="tozeroy",
        line=dict(color=THEME["accent_red"], width=1.5),
        fillcolor="rgba(248, 113, 113, 0.2)",
    ))
    fig_bt_dd.update_layout(yaxis_title="Drawdown (%)", height=400)
    st.plotly_chart(fig_bt_dd, use_container_width=True)

with tab_monthly:
    st.markdown("### Monthly Returns Table (%)")
    if not bt_res.monthly_returns_table.empty:
        st.dataframe(bt_res.monthly_returns_table, use_container_width=True)
    else:
        st.info("Insufficient duration to generate monthly returns table.")

with tab_pos:
    st.markdown("### Trading Turnover & Transaction Costs Paid")
    st.caption(f"Total Cumulative Turnover: {bt_res.total_turnover:,.2f}x Portfolio Value | Cumulative Costs & Slippage Paid: ${bt_res.total_costs_paid:,.2f}")
    st.dataframe(bt_res.positions_history.tail(30), use_container_width=True)
