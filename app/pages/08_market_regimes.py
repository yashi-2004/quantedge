"""
QuantEdge — Page 08: Market Regime Detection Laboratory
Gaussian Hidden Markov Models (HMM) and K-Means clustering.
Transition probability matrices, expected regime duration, and conditional return distributions.
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
from src.regimes.regime_detector import MarketRegimeDetector

st.set_page_config(page_title="Market Regimes | QuantEdge", page_icon="🌐", layout="wide")
apply_terminal_theme()

if "ingestion" not in st.session_state:
    st.session_state.ingestion = MarketDataIngestion()
ingestion: MarketDataIngestion = st.session_state.ingestion

st.markdown("## 🌐 Market Regime Detection Laboratory")
st.caption("Identify latent macroeconomic and volatility states using Gaussian Hidden Markov Models (HMM).")

c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    all_demo = DEMO_UNIVERSE["US"] + DEMO_UNIVERSE["INDIA"]
    ticker = st.selectbox("Market Asset", options=all_demo, index=0)
with c2:
    method = st.selectbox("Detection Method", options=["Gaussian HMM", "K-Means Clustering"], index=0)
with c3:
    n_regimes = st.selectbox("Number of Latent Regimes", options=[2, 3, 4], index=1)

res = ingestion.fetch_market_data(ticker)
df = res.df
returns = df["Close"].pct_change().dropna()
volatility = returns.rolling(21).std() * np.sqrt(252)

detector = MarketRegimeDetector()

with st.spinner("Estimating latent regime parameters..."):
    if method == "Gaussian HMM":
        reg_res = detector.detect_regimes_hmm(returns, volatility=volatility, n_regimes=n_regimes)
    else:
        reg_res = detector.detect_regimes_kmeans(returns, volatility, n_clusters=n_regimes)

# Summary Metrics
m_cols = st.columns(n_regimes)
for i in range(n_regimes):
    r_name = reg_res.regime_names[i]
    dur = reg_res.expected_durations_days.get(r_name, 0.0)
    with m_cols[i]:
        render_metric_card(f"{r_name[:18]}...", f"{dur} Days Dur.", f"State {i}")

# Asset Price Chart colored by Regime
st.markdown("### Regime Timeline Mapped to Asset Price Action")
fig_reg = create_plotly_figure()
price_aligned = df["Close"].loc[reg_res.regime_labels.index]

colors = [THEME["accent_green"], THEME["accent_blue"], THEME["accent_red"], THEME["accent_amber"]]
for k in range(n_regimes):
    mask = reg_res.regime_labels == k
    pts = price_aligned.copy()
    pts[~mask] = np.nan
    fig_reg.add_trace(go.Scatter(
        x=pts.index,
        y=pts,
        mode="markers+lines",
        name=reg_res.regime_names[k],
        line=dict(color=colors[k % len(colors)], width=2),
        marker=dict(size=3),
    ))

fig_reg.update_layout(yaxis_title=f"Price ({res.currency})", height=450)
st.plotly_chart(fig_reg, use_container_width=True)

# Transition Matrix & Conditional Metrics
col_trans, col_cond = st.columns(2)

with col_trans:
    st.markdown("### Regime Transition Probability Matrix (P_ij)")
    st.caption("Probability of transitioning from Regime i (row) to Regime j (column) in the next trading session.")
    fig_trans = px.imshow(
        reg_res.transition_matrix,
        text_auto=".2%",
        color_continuous_scale="Teal",
        template="plotly_dark",
    )
    fig_trans.update_layout(
        paper_bgcolor=THEME["bg_card"],
        plot_bgcolor=THEME["bg_secondary"],
        height=380,
        margin=dict(l=40, r=40, t=30, b=40),
    )
    st.plotly_chart(fig_trans, use_container_width=True)

with col_cond:
    st.markdown("### Conditional Return & Volatility by Regime")
    st.caption("Empirical annualized distribution statistics conditioned on each latent market state.")
    st.dataframe(reg_res.conditional_metrics, use_container_width=True)

st.info(f"**Diagnostic Summary:** {reg_res.summary_text}")
