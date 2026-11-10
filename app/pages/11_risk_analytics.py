"""
QuantEdge — Page 11: Risk Management Laboratory & Stress Testing
VaR (Historical, Parametric, Student-t, Monte Carlo), CVaR (Expected Shortfall),
Correlated Geometric Brownian Motion Monte Carlo simulation, and historical stress tests.
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
from src.risk.analytics import RiskLaboratory

st.set_page_config(page_title="Risk Analytics | QuantEdge", page_icon="🛡️", layout="wide")
apply_terminal_theme()

if "ingestion" not in st.session_state:
    st.session_state.ingestion = MarketDataIngestion()
ingestion: MarketDataIngestion = st.session_state.ingestion

st.markdown("## 🛡️ Risk Management Laboratory & Stress Testing")
st.caption("Tail risk quantification: Value at Risk (VaR), Expected Shortfall (CVaR), Monte Carlo paths, and macro stress tests.")

# Portfolio Selection
default_assets = ["SPY", "QQQ", "AAPL", "NVDA", "GLD", "TLT"]
selected_assets = st.multiselect("Active Portfolio Holdings", options=DEMO_UNIVERSE["US"] + DEMO_UNIVERSE["INDIA"], default=default_assets)

if len(selected_assets) < 1:
    st.warning("Please select at least 1 asset to perform risk analytics.")
    st.stop()

with st.sidebar:
    st.markdown("### 💼 Portfolio Sizing")
    port_capital = st.number_input("Portfolio Capital ($)", min_value=10_000, max_value=10_000_000, value=100_000, step=10_000)
    conf_level = st.selectbox("Confidence Level (1 - α)", options=[0.95, 0.99], index=0)
    horizon_days = st.selectbox("VaR Time Horizon", options=[1, 5, 10, 21], index=0, format_func=lambda x: f"{x}-Day Horizon")

# Load prices & portfolio returns (Equal Weighted as baseline or custom)
with st.spinner("Calculating portfolio historical returns..."):
    results = {t: ingestion.fetch_market_data(t) for t in selected_assets}
    close_df = ingestion.get_aligned_close_matrix(results)
    returns_df = close_df.pct_change().dropna()
    n_assets = len(selected_assets)
    weights = np.ones(n_assets) / n_assets
    port_returns = returns_df.dot(weights)

risk_lab = RiskLaboratory(portfolio_value=port_capital)

# Tabs
tab_var, tab_mc, tab_stress = st.tabs([
    "📉 Value at Risk & CVaR",
    "🎲 Monte Carlo Path Simulation",
    "💥 Macro Stress Testing",
])

with tab_var:
    st.markdown(f"### {int(conf_level*100)}% Value at Risk ({horizon_days}-Day Horizon)")
    var_rep = risk_lab.calculate_var_cvar(port_returns, confidence_level=conf_level, horizon_days=horizon_days)

    vm1, vm2, vm3, vm4 = st.columns(4)
    with vm1:
        render_metric_card(f"Historical VaR ({int(conf_level*100)}%)", f"{var_rep.historical_var_pct:.2f}%", f"${var_rep.dollar_var:,.0f} Max Loss")
    with vm2:
        render_metric_card(f"Expected Shortfall (CVaR)", f"{var_rep.historical_cvar_pct:.2f}%", f"${var_rep.dollar_cvar:,.0f} Tail Loss", is_positive=False)
    with vm3:
        render_metric_card("Parametric Gaussian VaR", f"{var_rep.parametric_var_pct:.2f}%")
    with vm4:
        render_metric_card("Student-t Fat-Tail VaR", f"{var_rep.student_t_var_pct:.2f}%")

    st.markdown("### VaR Distribution Threshold Plot")
    fig_var_dist = create_plotly_figure()
    fig_var_dist.add_trace(go.Histogram(
        x=port_returns * 100 * np.sqrt(horizon_days),
        nbinsx=70,
        name="Portfolio Returns",
        marker_color=THEME["accent_blue"],
        opacity=0.75,
    ))
    fig_var_dist.add_vline(
        x=-var_rep.historical_var_pct,
        line_dash="dash",
        line_color=THEME["accent_amber"],
        annotation_text=f"VaR ({var_rep.historical_var_pct:.2f}%)",
    )
    fig_var_dist.add_vline(
        x=-var_rep.historical_cvar_pct,
        line_dash="dash",
        line_color=THEME["accent_red"],
        annotation_text=f"CVaR ({var_rep.historical_cvar_pct:.2f}%)",
    )
    fig_var_dist.update_layout(xaxis_title="Portfolio Return (%)", yaxis_title="Frequency", height=400)
    st.plotly_chart(fig_var_dist, use_container_width=True)

with tab_mc:
    st.markdown("### Correlated Multi-Asset Geometric Brownian Motion Simulation")
    st.caption("Simulates 2,000 correlated asset return paths using Cholesky factorization of the empirical covariance matrix.")

    mc_horizon = st.slider("Forecast Simulation Horizon (Trading Days)", min_value=21, max_value=252, value=63, step=21)

    with st.spinner("Generating Monte Carlo paths..."):
        mc_res = risk_lab.monte_carlo_portfolio_simulation(returns_df, weights, horizon_days=mc_horizon, num_simulations=2000)

    mcm1, mcm2, mcm3 = st.columns(3)
    with mcm1:
        render_metric_card("Probability of Loss", f"{mc_res.probability_of_loss:.1f}%", is_positive=mc_res.probability_of_loss < 40)
    with mcm2:
        render_metric_card("Median Terminal Value", f"${mc_res.percentiles['50th Percentile (Median Outlook)']:,.0f}")
    with mcm3:
        render_metric_card("Severe Bear (5th %ile)", f"${mc_res.percentiles['5th Percentile (Severe Bear)']:,.0f}", is_positive=False)

    # Plot sample simulated paths
    fig_mc = create_plotly_figure()
    time_steps = list(range(mc_horizon + 1))
    for p_idx in range(min(60, len(mc_res.simulated_paths))):
        fig_mc.add_trace(go.Scatter(
            x=time_steps,
            y=mc_res.simulated_paths[p_idx],
            mode="lines",
            line=dict(width=1, color="rgba(56, 189, 248, 0.25)"),
            showlegend=False,
        ))
    fig_mc.add_hline(y=port_capital, line_dash="dash", line_color="#FFFFFF", annotation_text="Initial Capital")
    fig_mc.update_layout(xaxis_title="Days Forward", yaxis_title="Portfolio Value ($)", height=420)
    st.plotly_chart(fig_mc, use_container_width=True)

    st.markdown("#### Terminal Portfolio Value Quantiles")
    pct_df = pd.DataFrame(list(mc_res.percentiles.items()), columns=["Scenario", "Terminal Value ($)"]).set_index("Scenario")
    st.table(pct_df)

with tab_stress:
    st.markdown("### Historical & Hypothetical Macro Stress Scenarios")
    st.caption("Simulates immediate portfolio impact under historical black swan liquidity shocks and hypothetical crashes.")

    stress_scenario = st.selectbox(
        "Select Stress Event",
        options=[
            "2008 Global Financial Crisis",
            "2020 COVID Market Shock",
            "2022 Inflation & Rate Shock",
            "Hypothetical -25% Liquidity Crash",
        ],
        index=0,
    )

    stress_res = risk_lab.run_stress_test(
        pd.Series(weights, index=selected_assets),
        scenario_name=stress_scenario,
        custom_market_shock=-0.25 if "Hypothetical" in stress_scenario else None,
    )

    stm1, stm2, stm3 = st.columns(3)
    with stm1:
        render_metric_card("Portfolio Impact", f"{stress_res.portfolio_impact_pct:+.2f}%", is_positive=stress_res.portfolio_impact_pct >= 0)
    with stm2:
        render_metric_card("Estimated Capital Impact", f"${stress_res.dollar_loss:,.2f}", is_positive=stress_res.dollar_loss >= 0)
    with stm3:
        render_metric_card("Scenario Nature", "HYPOTHETICAL" if stress_res.is_hypothetical else "HISTORICAL")

    st.markdown("#### Individual Asset Shocks in Selected Event")
    shocks_df = pd.DataFrame(list(stress_res.asset_shocks.items()), columns=["Asset", "Shock (%)"]).set_index("Asset")
    st.dataframe(shocks_df, use_container_width=True)
