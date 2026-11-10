"""
QuantEdge — Page 03: Financial Feature Engineering Lab
Extracts technical indicators, statistical features, range volatility estimators,
and evaluates correlation with forward returns. Zero lookahead leakage.
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
from src.data.ingestion import DEMO_UNIVERSE, MarketDataIngestion
from src.features.pipeline import FeatureEngine

st.set_page_config(page_title="Feature Lab | QuantEdge", page_icon="⚙️", layout="wide")
apply_terminal_theme()

if "ingestion" not in st.session_state:
    st.session_state.ingestion = MarketDataIngestion()
ingestion: MarketDataIngestion = st.session_state.ingestion

st.markdown("## ⚙️ Financial Feature Engineering Lab")
st.caption("Point-in-time quantitative feature generation, volatility surfaces, and mutual information ranking.")

c1, c2 = st.columns([2, 1])
with c1:
    all_demo = DEMO_UNIVERSE["US"] + DEMO_UNIVERSE["INDIA"]
    ticker = st.selectbox("Select Target Instrument", options=all_demo, index=0)
with c2:
    inc_macro = st.checkbox("Include Macro Proxies (Rates & VIX)", value=True)

with st.spinner("Generating feature universe..."):
    res = ingestion.fetch_market_data(ticker)
    bench_res = ingestion.fetch_market_data("SPY") if ticker != "SPY" else None
    bench_df = bench_res.df if bench_res else None

    engine = FeatureEngine(include_macro=inc_macro)
    fmat = engine.build_feature_matrix(res.df, benchmark_df=bench_df)

# Feature Matrix Summary
m1, m2, m3, m4 = st.columns(4)
with m1:
    render_metric_card("Total Engineered Features", f"{len(fmat.feature_names)}")
with m2:
    render_metric_card("Aligned Sample Rows", f"{len(fmat.X):,}")
with m3:
    render_metric_card("Highest Corr Feature", f"{fmat.correlation_with_target.index[0]}")
with m4:
    max_c = fmat.correlation_with_target.iloc[0]
    render_metric_card("Max Target Correlation", f"{max_c:+.4f}")

# Tabs: Indicators, Volatility, Feature Importance
tab_indicators, tab_vols, tab_ranking, tab_matrix = st.tabs([
    "📊 Technical Indicators",
    "⚡ Range Volatility Estimators",
    "🎯 Feature Target Ranking",
    "📑 Feature Matrix Explorer",
])

with tab_indicators:
    st.markdown("### Technical Indicators Visualization")
    ind_choice = st.selectbox("Select Indicator to Plot", options=["RSI_14", "MACD", "Bollinger Bands"], index=0)

    if ind_choice == "RSI_14":
        fig_rsi = create_plotly_figure()
        fig_rsi.add_trace(go.Scatter(x=fmat.X.index, y=fmat.X["RSI_14"], name="RSI (14)", line=dict(color=THEME["accent_teal"])))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color=THEME["accent_red"], annotation_text="Overbought (70)")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color=THEME["accent_green"], annotation_text="Oversold (30)")
        fig_rsi.update_layout(yaxis_title="RSI Index", height=400)
        st.plotly_chart(fig_rsi, use_container_width=True)

    elif ind_choice == "MACD":
        fig_macd = create_plotly_figure()
        fig_macd.add_trace(go.Scatter(x=fmat.X.index, y=fmat.X["MACD_12_26"], name="MACD Line", line=dict(color=THEME["accent_blue"])))
        fig_macd.add_trace(go.Scatter(x=fmat.X.index, y=fmat.X["MACD_Signal_9"], name="Signal Line", line=dict(color=THEME["accent_amber"])))
        fig_macd.add_trace(go.Bar(x=fmat.X.index, y=fmat.X["MACD_Hist"], name="Histogram", marker_color=THEME["accent_teal"]))
        fig_macd.update_layout(yaxis_title="MACD Value", height=400)
        st.plotly_chart(fig_macd, use_container_width=True)

    elif ind_choice == "Bollinger Bands":
        fig_bb = create_plotly_figure()
        price_s = res.df["Close"].loc[fmat.X.index]
        fig_bb.add_trace(go.Scatter(x=fmat.X.index, y=price_s, name="Close Price", line=dict(color="#FFFFFF", width=1.5)))
        fig_bb.add_trace(go.Scatter(x=fmat.X.index, y=fmat.X["BB_Upper_20"], name="Upper Band (+2σ)", line=dict(color=THEME["accent_red"], dash="dash")))
        fig_bb.add_trace(go.Scatter(x=fmat.X.index, y=fmat.X["BB_Mid_20"], name="Middle SMA (20)", line=dict(color=THEME["accent_blue"], dash="dot")))
        fig_bb.add_trace(go.Scatter(x=fmat.X.index, y=fmat.X["BB_Lower_20"], name="Lower Band (-2σ)", line=dict(color=THEME["accent_green"], dash="dash")))
        fig_bb.update_layout(yaxis_title=f"Price ({res.currency})", height=420)
        st.plotly_chart(fig_bb, use_container_width=True)

with tab_vols:
    st.markdown("### Advanced Volatility Estimator Comparison")
    st.caption("Compares Close-to-Close realized volatility with Parkinson High-Low and Garman-Klass OHLC estimators.")
    
    fig_vols = create_plotly_figure()
    fig_vols.add_trace(go.Scatter(x=fmat.X.index, y=fmat.X["Vol_Realized_21"] * 100, name="Realized Close-to-Close (21d)", line=dict(color=THEME["accent_blue"])))
    fig_vols.add_trace(go.Scatter(x=fmat.X.index, y=fmat.X["Vol_Parkinson_21"] * 100, name="Parkinson High-Low (21d)", line=dict(color=THEME["accent_teal"], width=2)))
    fig_vols.add_trace(go.Scatter(x=fmat.X.index, y=fmat.X["Vol_Garman_Klass_21"] * 100, name="Garman-Klass OHLC (21d)", line=dict(color=THEME["accent_amber"], width=1.5)))
    fig_vols.update_layout(yaxis_title="Annualized Volatility (%)", height=420)
    st.plotly_chart(fig_vols, use_container_width=True)

with tab_ranking:
    st.markdown("### Feature-to-Target Correlation Ranking")
    st.caption("Univariate Pearson correlation between each feature at time t and forward return R_{t+1}.")
    
    top_corr = fmat.correlation_with_target.head(15)
    fig_corr_bar = px.bar(
        x=top_corr.values,
        y=top_corr.index,
        orientation="h",
        color=top_corr.values,
        color_continuous_scale="Teal",
        template="plotly_dark",
    )
    fig_corr_bar.update_layout(
        paper_bgcolor=THEME["bg_card"],
        plot_bgcolor=THEME["bg_secondary"],
        xaxis_title="Pearson Correlation with Target R_{t+1}",
        yaxis_title="Feature",
        height=450,
    )
    st.plotly_chart(fig_corr_bar, use_container_width=True)

with tab_matrix:
    st.markdown("### Feature Matrix Snapshot (Last 50 Rows)")
    st.dataframe(fmat.X.tail(50), use_container_width=True)
