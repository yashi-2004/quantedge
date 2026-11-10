"""
QuantEdge — Page 12: Reinforcement Learning Allocation Laboratory
Gymnasium-compatible portfolio allocation environment, Actor-Critic (A2C) policy gradients,
and out-of-sample benchmark evaluation.
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
from src.rl.agent import RLPortfolioTrainer

st.set_page_config(page_title="Reinforcement Learning | QuantEdge", page_icon="🕹️", layout="wide")
apply_terminal_theme()

if "ingestion" not in st.session_state:
    st.session_state.ingestion = MarketDataIngestion()
ingestion: MarketDataIngestion = st.session_state.ingestion

st.markdown("## 🕹️ Reinforcement Learning Allocation Laboratory")
st.caption("Gymnasium-compliant sequential portfolio allocation via policy gradient Actor-Critic networks.")

# Holdings Universe
default_universe = ["SPY", "QQQ", "AAPL", "NVDA", "GLD"]
selected_assets = st.multiselect("Asset Pool", options=DEMO_UNIVERSE["US"] + DEMO_UNIVERSE["INDIA"], default=default_universe)

if len(selected_assets) < 2:
    st.warning("Please select at least 2 assets for the RL agent to allocate across.")
    st.stop()

c1, c2, c3 = st.columns(3)
with c1:
    episodes = st.slider("Training Episodes", min_value=10, max_value=40, value=20, step=5)
with c2:
    window_size = st.slider("State Observation Lookback Window", min_value=10, max_value=30, value=15)
with c3:
    lr = st.select_slider("Actor-Critic Learning Rate", options=[0.001, 0.003, 0.005, 0.01], value=0.003)

# Load data and partition chronologically
with st.spinner("Preparing sequential market environment..."):
    results = {t: ingestion.fetch_market_data(t) for t in selected_assets}
    close_df = ingestion.get_aligned_close_matrix(results)
    returns_df = close_df.pct_change().dropna()

    # Chronological 75% train / 25% test split
    split_idx = int(len(returns_df) * 0.75)
    train_ret = returns_df.iloc[:split_idx]
    test_ret = returns_df.iloc[split_idx:]

st.markdown(f"**Chronological Environment:** Training ({len(train_ret)} steps) | Out-of-Sample Test ({len(test_ret)} steps: {test_ret.index[0].date()} to {test_ret.index[-1].date()})")

if st.button("🚀 Train Policy Gradient RL Agent"):
    with st.spinner("Training Actor-Critic Agent on CPU..."):
        trainer = RLPortfolioTrainer(episodes=episodes, lr=lr)
        rl_res = trainer.train_and_evaluate(train_ret, test_ret, window_size=window_size)

    # Top Metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_card("RL Test Return", f"{rl_res.test_total_return:+.1f}%", is_positive=rl_res.test_total_return >= 0)
    with m2:
        render_metric_card("RL Out-of-Sample Sharpe", f"{rl_res.test_sharpe:.2f}", is_positive=rl_res.test_sharpe >= 1.0)
    with m3:
        render_metric_card("RL Max Drawdown", f"{rl_res.test_max_drawdown:.2f}%", is_positive=False)
    with m4:
        render_metric_card("Total Training Episodes", f"{len(rl_res.episode_rewards)}")

    # Visualizations
    col_curve, col_weights = st.columns(2)

    with col_curve:
        st.markdown("### Training Episode Cumulative Reward Progression")
        fig_r = create_plotly_figure()
        fig_r.add_trace(go.Scatter(
            x=list(range(1, len(rl_res.episode_rewards) + 1)),
            y=rl_res.episode_rewards,
            mode="lines+markers",
            name="Episode Reward",
            line=dict(color=THEME["accent_teal"], width=2),
        ))
        fig_r.update_layout(xaxis_title="Episode", yaxis_title="Cumulative Reward", height=380)
        st.plotly_chart(fig_r, use_container_width=True)

    with col_weights:
        st.markdown("### Out-of-Sample Dynamic Asset Weights")
        fig_w = px.area(
            rl_res.test_weights,
            template="plotly_dark",
            labels={"value": "Weight", "index": "Date"},
        )
        fig_w.update_layout(
            paper_bgcolor=THEME["bg_card"],
            plot_bgcolor=THEME["bg_secondary"],
            height=380,
            yaxis_title="Allocation Weight",
        )
        st.plotly_chart(fig_w, use_container_width=True)

    # Benchmark Comparison Table
    st.markdown("### Out-of-Sample RL Agent vs Baseline Comparison")
    st.dataframe(rl_res.benchmark_comparison, use_container_width=True)

    st.warning(
        "⚠️ **Research Methodology Disclosure:** Reinforcement learning in financial markets can easily overfit historical trajectories. "
        "High in-sample episode rewards do not guarantee generalization out-of-sample under non-stationary market regimes."
    )
