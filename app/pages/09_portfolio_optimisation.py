"""
QuantEdge — Page 09: Portfolio Optimisation & Asset Allocation Engine
Markowitz Mean-Variance, Risk Parity, Black-Litterman, CVaR Optimisation,
Efficient Frontier, and Risk Decomposition (MCR/PCR).
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
from src.portfolio.optimisation import PortfolioOptimizer

st.set_page_config(page_title="Portfolio Optimisation | QuantEdge", page_icon="⚖️", layout="wide")
apply_terminal_theme()

if "ingestion" not in st.session_state:
    st.session_state.ingestion = MarketDataIngestion()
ingestion: MarketDataIngestion = st.session_state.ingestion

st.markdown("## ⚖️ Portfolio Optimisation & Allocation Engine")
st.caption("Convex portfolio construction: Markowitz Mean-Variance, Risk Parity, Black-Litterman, and CVaR minimization.")

# Asset Selection
default_universe = ["SPY", "QQQ", "AAPL", "NVDA", "GLD", "TLT"]
selected_assets = st.multiselect(
    "Portfolio Universe", options=DEMO_UNIVERSE["US"] + DEMO_UNIVERSE["INDIA"], default=default_universe
)

if len(selected_assets) < 2:
    st.warning("Please select at least 2 assets to perform portfolio optimization.")
    st.stop()

# Load data and returns
with st.spinner("Fetching asset history and computing covariance..."):
    results = {t: ingestion.fetch_market_data(t) for t in selected_assets}
    close_df = ingestion.get_aligned_close_matrix(results)
    returns_df = close_df.pct_change().dropna()

optimizer = PortfolioOptimizer(risk_free_rate=0.03)

# Sidebar Constraints
with st.sidebar:
    st.markdown("### 🔧 Allocation Constraints")
    max_w = st.slider("Maximum Asset Weight Constraint", min_value=0.20, max_value=1.0, value=0.50, step=0.05)
    rf_rate = st.number_input("Risk-Free Rate (Annualized)", min_value=0.0, max_value=0.10, value=0.03, step=0.005)
    optimizer.rf = rf_rate

# Compute All 5 Allocation Models
res_ew = optimizer.equal_weighted(returns_df)
res_ms = optimizer.max_sharpe(returns_df, max_weight=max_w)
res_mv = optimizer.minimum_variance(returns_df, max_weight=max_w)
res_rp = optimizer.risk_parity(returns_df, max_weight=max_w)
res_cvar = optimizer.min_cvar(returns_df, alpha=0.95, max_weight=max_w)

# Interactive Model Selection
tab_compare, tab_frontier, tab_bl, tab_decomp = st.tabs([
    "📊 Allocation Comparison",
    "📈 Markowitz Efficient Frontier",
    "🧠 Black-Litterman Model",
    "🔬 Risk Decomposition (MCR & PCR)",
])

with tab_compare:
    st.markdown("### Model Comparison Matrix")
    all_models = [res_ew, res_ms, res_mv, res_rp, res_cvar]

    summary_rows = [
        {
            "Strategy": m.method,
            "Expected Return (%)": round(m.expected_return_annual * 100.0, 2),
            "Annual Volatility (%)": round(m.volatility_annual * 100.0, 2),
            "Sharpe Ratio": m.sharpe_ratio,
            "Sortino Ratio": m.sortino_ratio,
            "95% CVaR (%)": round(m.cvar_95 * 100.0, 2),
            "Diversification Ratio": m.diversification_ratio,
        }
        for m in all_models
    ]
    st.dataframe(pd.DataFrame(summary_rows).set_index("Strategy"), use_container_width=True)

    # Weights Bar Chart across all models
    st.markdown("### Asset Weights Distribution Across Optimization Frameworks")
    weights_combined = pd.DataFrame({
        "Equal Weight": res_ew.weights * 100,
        "Max Sharpe": res_ms.weights * 100,
        "Min Variance": res_mv.weights * 100,
        "Risk Parity": res_rp.weights * 100,
        "Min CVaR": res_cvar.weights * 100,
    })

    fig_w = px.bar(
        weights_combined.T,
        barmode="stack",
        template="plotly_dark",
        labels={"value": "Weight (%)", "index": "Strategy"},
    )
    fig_w.update_layout(
        paper_bgcolor=THEME["bg_card"],
        plot_bgcolor=THEME["bg_secondary"],
        height=400,
        yaxis_title="Allocation (%)",
    )
    st.plotly_chart(fig_w, use_container_width=True)

with tab_frontier:
    st.markdown("### Tracing the Markowitz Efficient Frontier")
    st.caption("Hyperbolic boundary of maximum expected return for every level of portfolio risk.")

    ef_res = optimizer.compute_efficient_frontier(returns_df, num_points=35)
    valid_mask = ~np.isnan(ef_res.target_volatilities)

    fig_ef = create_plotly_figure()
    # Frontier line
    fig_ef.add_trace(go.Scatter(
        x=ef_res.target_volatilities[valid_mask] * 100,
        y=ef_res.target_returns[valid_mask] * 100,
        mode="lines",
        name="Efficient Frontier",
        line=dict(color=THEME["accent_teal"], width=3),
    ))
    # Individual Assets
    asset_means = returns_df.mean() * 252 * 100
    asset_vols = returns_df.std() * np.sqrt(252) * 100
    fig_ef.add_trace(go.Scatter(
        x=asset_vols,
        y=asset_means,
        mode="markers+text",
        text=asset_means.index,
        textposition="top center",
        name="Individual Securities",
        marker=dict(color="#A0AEC0", size=10, symbol="diamond"),
    ))
    # Optimal Portfolios Highlight
    fig_ef.add_trace(go.Scatter(
        x=[res_ms.volatility_annual * 100],
        y=[res_ms.expected_return_annual * 100],
        mode="markers+text",
        text=["Max Sharpe"],
        textposition="bottom right",
        name="Max Sharpe Portfolio",
        marker=dict(color=THEME["accent_amber"], size=14, symbol="star"),
    ))
    fig_ef.add_trace(go.Scatter(
        x=[res_mv.volatility_annual * 100],
        y=[res_mv.expected_return_annual * 100],
        mode="markers+text",
        text=["Min Vol"],
        textposition="bottom right",
        name="Minimum Variance",
        marker=dict(color=THEME["accent_blue"], size=12, symbol="circle"),
    ))

    fig_ef.update_layout(xaxis_title="Annualized Volatility (%)", yaxis_title="Annualized Return (%)", height=450)
    st.plotly_chart(fig_ef, use_container_width=True)

with tab_bl:
    st.markdown("### Black-Litterman Bayesian Allocation")
    st.caption("Combines reverse-optimized market equilibrium prior with investor subjective views.")

    bl_c1, bl_c2 = st.columns(2)
    with bl_c1:
        view_asset = st.selectbox("Select Asset to Express View On", options=selected_assets, index=0)
    with bl_c2:
        view_ret = st.slider(f"Expected Annual Return View for {view_asset} (%)", min_value=-20.0, max_value=50.0, value=15.0, step=1.0)

    res_bl = optimizer.black_litterman(returns_df, views={view_asset: view_ret / 100.0}, tau=0.05)

    bl_m1, bl_m2, bl_m3 = st.columns(3)
    with bl_m1:
        render_metric_card("BL Expected Return", f"{res_bl.expected_return_annual*100:.2f}%")
    with bl_m2:
        render_metric_card("BL Annual Volatility", f"{res_bl.volatility_annual*100:.2f}%")
    with bl_m3:
        render_metric_card("BL Sharpe Ratio", f"{res_bl.sharpe_ratio:.2f}")

    fig_bl = px.pie(
        names=res_bl.weights.index,
        values=res_bl.weights.values * 100,
        title="Black-Litterman Posterior Allocation",
        template="plotly_dark",
    )
    fig_bl.update_layout(paper_bgcolor=THEME["bg_card"], plot_bgcolor=THEME["bg_secondary"], height=380)
    st.plotly_chart(fig_bl, use_container_width=True)

with tab_decomp:
    st.markdown("### Risk Decomposition: Marginal & Percentage Contribution to Risk")
    st.caption("Examines how much each individual security contributes to total portfolio risk variance.")

    decomp_choice = st.selectbox("Inspect Strategy Decomposition", options=["Risk Parity", "Max Sharpe", "Equal Weight"], index=0)
    target_res = res_rp if decomp_choice == "Risk Parity" else (res_ms if decomp_choice == "Max Sharpe" else res_ew)

    col_mcr, col_pcr = st.columns(2)
    with col_mcr:
        st.markdown("#### Marginal Contribution to Risk (∂σ_p / ∂w_i)")
        st.bar_chart(target_res.marginal_risk_contributions)
    with col_pcr:
        st.markdown("#### Percentage Contribution to Risk (% PCR)")
        st.bar_chart(target_res.percentage_risk_contributions)
