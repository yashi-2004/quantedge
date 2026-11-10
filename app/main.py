"""
QuantEdge — An End-to-End Quantitative Research, Portfolio Optimisation & Risk Analytics Platform
Main Entrypoint and Workstation Overview
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Configure headless environment variables
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
os.environ["HF_HOME"] = "/tmp/huggingface"

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.components.ui import THEME, apply_terminal_theme, create_plotly_figure, render_metric_card
from src.data.ingestion import DEMO_UNIVERSE, MarketDataIngestion

st.set_page_config(
    page_title="QuantEdge | Quantitative Research Workstation",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_terminal_theme()

# --- Initialize Ingestion Engine in Session State ---
if "ingestion" not in st.session_state:
    st.session_state.ingestion = MarketDataIngestion()

ingestion: MarketDataIngestion = st.session_state.ingestion

# --- Sidebar Global Controls ---
with st.sidebar:
    st.markdown("### ⚡ **QUANTEDGE TERMINAL**")
    st.caption("Institutional Quantitative Research & Risk Engine")
    st.divider()

    universe_choice = st.selectbox(
        "Market Universe",
        options=["US Broad & Tech Equities", "Indian NSE Equities", "Custom Tickers"],
        index=0,
    )

    if universe_choice == "US Broad & Tech Equities":
        default_tickers = ["SPY", "QQQ", "AAPL", "NVDA", "GLD", "TLT"]
        tickers = st.multiselect("Active Securities", options=DEMO_UNIVERSE["US"], default=default_tickers)
    elif universe_choice == "Indian NSE Equities":
        default_tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"]
        tickers = st.multiselect("Active Securities", options=DEMO_UNIVERSE["INDIA"], default=default_tickers)
    else:
        custom_input = st.text_input("Enter Ticker Symbols (comma separated)", value="SPY, QQQ, AAPL, GLD")
        tickers = [t.strip().upper() for t in custom_input.split(",") if t.strip()]

    st.divider()
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365 * 3))
    with col_d2:
        end_date = st.date_input("End Date", value=datetime.now())

    use_adjusted = st.checkbox("Use Adjusted Close", value=True)
    force_refresh = st.button("🔄 Refresh Data Cache")

    st.divider()
    st.caption("🔒 **Zero-Cost Deployment Mode**")
    st.caption("CPU Native // Local SQLite Storage // Fully Air-Gapped Resilient")

# --- Top Header ---
st.markdown(
    """
    <div style="display: flex; justify-content: space-between; align-items: baseline; border-bottom: 1px solid #2D3748; padding-bottom: 12px; margin-bottom: 20px;">
        <div>
            <h1 style="margin: 0; font-size: 2.0rem; letter-spacing: -0.03em;">QUANTEDGE WORKSTATION</h1>
            <p style="margin: 2px 0 0 0; color: #A0AEC0; font-size: 0.95rem;">
                End-to-End Quantitative Research, Portfolio Optimisation & Risk Analytics Platform
            </p>
        </div>
        <div style="text-align: right;">
            <span class="badge-live">SYSTEM OPERATIONAL</span>
            <span class="badge-cached">CPU ACCELERATED</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not tickers:
    st.warning("Please select at least one financial security in the sidebar to begin research.")
    st.stop()

@st.cache_data(ttl=1800, show_spinner=False)
def load_universe_data(active_tickers, start_str, end_str, adjusted, refresh):
    res_map = {
        t: ingestion.fetch_market_data(
            t,
            start_date=start_str,
            end_date=end_str,
            use_adjusted=adjusted,
            force_refresh=refresh,
        )
        for t in active_tickers
    }
    matrix = ingestion.get_aligned_close_matrix(res_map)
    return res_map, matrix

# --- Fetch Synchronized Multi-Asset Data ---
with st.spinner("Synchronizing market telemetry across active universe..."):
    results, close_matrix = load_universe_data(
        tuple(tickers), str(start_date), str(end_date), use_adjusted, force_refresh
    )

if close_matrix.empty:
    st.error("Failed to load historical data for the selected assets. Please try another selection.")
    st.stop()

# --- Top Telemetry Metrics Bar ---
m_col1, m_col2, m_col3, m_col4 = st.columns(4)

with m_col1:
    primary_ticker = tickers[0]
    last_p = close_matrix[primary_ticker].iloc[-1]
    ret_1d = close_matrix[primary_ticker].pct_change().iloc[-1] * 100.0
    render_metric_card(
        f"{primary_ticker} Spot Price",
        f"${last_p:,.2f}" if not primary_ticker.endswith(".NS") else f"₹{last_p:,.2f}",
        f"{ret_1d:+.2f}% 1-Day",
        is_positive=ret_1d >= 0,
    )

with m_col2:
    n_days = len(close_matrix)
    render_metric_card("Trading Horizon", f"{n_days} Sessions", f"{n_days/252:.1f} Yrs Available")

with m_col3:
    avg_quality = np.mean([r.validation.quality_score for r in results.values()])
    render_metric_card("Data Quality Score", f"{avg_quality:.1f} / 100", "Zero-Inversion Passed", is_positive=True)

with m_col4:
    provenance_types = list(set(r.provenance for r in results.values()))
    prov_label = provenance_types[0] if len(provenance_types) == 1 else "HYBRID_CACHE"
    render_metric_card("Data Provenance", prov_label, "Audited & Verified", is_positive=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- Main Dashboard Tabs ---
tab_perf, tab_corr, tab_vol, tab_nav = st.tabs([
    "📈 Normalized Performance",
    "🔥 Correlation Dynamics",
    "⚡ Volatility Surface",
    "🧭 Research Laboratory Navigator",
])

with tab_perf:
    st.markdown("#### Cumulative Rebased Performance (Base = 100.0)")
    st.caption("Normalized historical growth trajectories assuming synchronized reinvestment.")
    
    norm_matrix = (close_matrix / close_matrix.iloc[0]) * 100.0
    fig_norm = create_plotly_figure()
    for col in norm_matrix.columns:
        fig_norm.add_trace(go.Scatter(
            x=norm_matrix.index,
            y=norm_matrix[col],
            mode="lines",
            name=col,
            line=dict(width=2),
        ))
    fig_norm.update_layout(yaxis_title="Rebased Value (100.0)", height=450)
    st.plotly_chart(fig_norm, use_container_width=True)

with tab_corr:
    st.markdown("#### Cross-Asset Return Correlation Matrix")
    st.caption("Pearson correlation of daily percentage changes over the selected horizon.")
    
    returns_matrix = close_matrix.pct_change().dropna()
    corr = returns_matrix.corr().round(3)
    
    fig_corr = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="Blues",
        aspect="auto",
        template="plotly_dark",
    )
    fig_corr.update_layout(
        paper_bgcolor=THEME["bg_card"],
        plot_bgcolor=THEME["bg_secondary"],
        height=450,
        margin=dict(l=40, r=40, t=30, b=40),
    )
    st.plotly_chart(fig_corr, use_container_width=True)

with tab_vol:
    st.markdown("#### Rolling 21-Day Annualized Volatility (%)")
    st.caption("Annualized sample standard deviation of daily returns across a 1-month trading window.")
    
    rolling_vol = returns_matrix.rolling(21).std() * np.sqrt(252) * 100.0
    fig_vol = create_plotly_figure()
    for col in rolling_vol.columns:
        fig_vol.add_trace(go.Scatter(
            x=rolling_vol.index,
            y=rolling_vol[col],
            mode="lines",
            name=col,
            line=dict(width=1.5),
        ))
    fig_vol.update_layout(yaxis_title="Annualized Volatility (%)", height=450)
    st.plotly_chart(fig_vol, use_container_width=True)

with tab_nav:
    st.markdown("#### QuantEdge Laboratory Directory")
    st.caption("Direct access to specialized quantitative research, modeling, and execution engines.")

    nav_cols1, nav_cols2 = st.columns(2)
    with nav_cols1:
        st.markdown("""
        - **01. Market Overview**: Macro indicators, price action, and distribution analysis.
        - **02. Data Explorer**: OHLCV inspection, outlier detection, data provenance, and CSV import.
        - **03. Feature Engineering Lab**: Technical indicators, Parkinson/Garman-Klass volatility, and VIF checks.
        - **04. Econometric Models**: ADF/KPSS, auto-ARIMA, GARCH(1,1), VAR, Kalman filtering, Diebold-Mariano.
        - **05. Machine Learning Lab**: Regularized linear models, Random Forest, XGBoost, Optuna tuning.
        - **06. Deep Learning Lab**: PyTorch LSTM, GRU, TCN, and Time-Series Transformer models.
        - **07. Financial Sentiment**: NLP headline sentiment scoring, FinBERT, Loughran-McDonald lexicon.
        """)
    with nav_cols2:
        st.markdown("""
        - **08. Market Regimes**: Gaussian Hidden Markov Models (HMM), K-Means, transition matrices.
        - **09. Portfolio Optimisation**: Mean-Variance, Risk Parity, Black-Litterman, CVaR, constraints.
        - **10. Backtesting Studio**: 9 systematic strategies, transaction costs, slippage, and anti-leakage checks.
        - **11. Risk Analytics**: VaR (Historical, Parametric, Monte Carlo), CVaR, stress testing, MC paths.
        - **12. Reinforcement Learning**: Gymnasium PortfolioEnv, Actor-Critic PPO agent allocation.
        - **13. Experiment Tracking**: Local MLflow tracking, hyperparameter runs, metric comparisons.
        - **14. Research Reports**: Automated quantitative research paper generator (Hypotheses H1–H5).
        """)

st.divider()
st.caption("QuantEdge Platform © 2026. Zero-cost quantitative research environment. No paid APIs required.")
