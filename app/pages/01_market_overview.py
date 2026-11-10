"""
QuantEdge — Page 01: Market Overview & Asset Telemetry
Comprehensive exploratory financial analysis, candlestick charts, distributions, and drawdowns.
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
import streamlit as st

from app.components.ui import THEME, apply_terminal_theme, create_plotly_figure, render_metric_card
from src.data.ingestion import DEMO_UNIVERSE, MarketDataIngestion

st.set_page_config(page_title="Market Overview | QuantEdge", page_icon="📊", layout="wide")
apply_terminal_theme()

if "ingestion" not in st.session_state:
    st.session_state.ingestion = MarketDataIngestion()
ingestion: MarketDataIngestion = st.session_state.ingestion

st.markdown("## 📊 Market Overview & Asset Telemetry")
st.caption("Deep historical inspection of individual financial instruments and return distributions.")

# Controls
c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    all_demo = DEMO_UNIVERSE["US"] + DEMO_UNIVERSE["INDIA"]
    ticker = st.selectbox("Select Security", options=all_demo, index=0)
with c2:
    lookback_years = st.selectbox("Horizon", options=[1, 2, 3, 5], index=2)
with c3:
    use_adj = st.checkbox("Adjusted Prices", value=True)

start_date = str((datetime.now() - timedelta(days=365 * lookback_years)).date())

with st.spinner(f"Retrieving data for {ticker}..."):
    res = ingestion.fetch_market_data(ticker, start_date=start_date, use_adjusted=use_adj)
    df = res.df

if df.empty:
    st.error(f"Could not load data for {ticker}.")
    st.stop()

# Key metrics
ret = df["Close"].pct_change().dropna()
ann_return = ret.mean() * 252 * 100.0
ann_vol = ret.std() * np.sqrt(252) * 100.0
sharpe = (ann_return - 3.0) / (ann_vol + 1e-12)

roll_max = df["Close"].cummax()
dd = ((df["Close"] - roll_max) / roll_max) * 100.0
max_dd = dd.min()

m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    render_metric_card("Last Price", f"{df['Close'].iloc[-1]:,.2f} {res.currency}")
with m2:
    render_metric_card("Annualized Return", f"{ann_return:+.2f}%", is_positive=ann_return >= 0)
with m3:
    render_metric_card("Annualized Volatility", f"{ann_vol:.2f}%")
with m4:
    render_metric_card("Sharpe Ratio (Rf=3%)", f"{sharpe:.2f}", is_positive=sharpe >= 1.0)
with m5:
    render_metric_card("Max Historical Drawdown", f"{max_dd:.2f}%", is_positive=False)

# Candlestick with Volume
st.markdown("### Interactive Candlestick Price Action")
fig_candle = go.Figure()
fig_candle.add_trace(go.Candlestick(
    x=df.index,
    open=df["Open"],
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
    name="OHLC",
    increasing_line_color=THEME["accent_green"],
    decreasing_line_color=THEME["accent_red"],
))
fig_candle.update_layout(
    template="plotly_dark",
    paper_bgcolor=THEME["bg_card"],
    plot_bgcolor=THEME["bg_secondary"],
    height=500,
    xaxis_rangeslider_visible=False,
    yaxis_title=f"Price ({res.currency})",
    margin=dict(l=40, r=40, t=20, b=40),
)
st.plotly_chart(fig_candle, use_container_width=True)

# Two columns: Distribution and Underwater Drawdown
col_dist, col_dd = st.columns(2)

with col_dist:
    st.markdown("### Daily Return Distribution")
    fig_dist = create_plotly_figure()
    fig_dist.add_trace(go.Histogram(
        x=ret * 100.0,
        nbinsx=60,
        name="Empirical Returns",
        marker_color=THEME["accent_blue"],
        opacity=0.75,
        histnorm="probability density",
    ))
    
    # Overlay Normal distribution curve
    x_grid = np.linspace(ret.min() * 100.0, ret.max() * 100.0, 100)
    pdf = stats.norm.pdf(x_grid, loc=ret.mean() * 100.0, scale=ret.std() * 100.0)
    fig_dist.add_trace(go.Scatter(
        x=x_grid,
        y=pdf,
        mode="lines",
        name="Gaussian Fit",
        line=dict(color=THEME["accent_amber"], width=2, dash="dash"),
    ))
    fig_dist.update_layout(xaxis_title="Daily Return (%)", yaxis_title="Density", height=380)
    st.plotly_chart(fig_dist, use_container_width=True)

with col_dd:
    st.markdown("### Underwater Drawdown Profile")
    fig_dd = create_plotly_figure()
    fig_dd.add_trace(go.Scatter(
        x=dd.index,
        y=dd,
        mode="lines",
        name="Drawdown",
        fill="tozeroy",
        line=dict(color=THEME["accent_red"], width=1.5),
        fillcolor="rgba(248, 113, 113, 0.2)",
    ))
    fig_dd.update_layout(xaxis_title="Date", yaxis_title="Drawdown (%)", height=380)
    st.plotly_chart(fig_dd, use_container_width=True)

# Statistical Summary Table
st.markdown("### Empirical Return Moments & Statistics")
stats_data = {
    "Metric": [
        "Observations (Trading Days)",
        "Mean Daily Return (%)",
        "Median Daily Return (%)",
        "Daily Volatility (%)",
        "Skewness (Fisher-Pearson)",
        "Excess Kurtosis",
        "Best Single Session (%)",
        "Worst Single Session (%)",
        "Positive Sessions (%)",
    ],
    "Value": [
        f"{len(ret)}",
        f"{ret.mean()*100:.4f}%",
        f"{ret.median()*100:.4f}%",
        f"{ret.std()*100:.4f}%",
        f"{stats.skew(ret):.3f}",
        f"{stats.kurtosis(ret):.3f}",
        f"{ret.max()*100:+.2f}%",
        f"{ret.min()*100:+.2f}%",
        f"{(ret > 0).mean()*100:.1f}%",
    ],
}
st.table(pd.DataFrame(stats_data).set_index("Metric"))
