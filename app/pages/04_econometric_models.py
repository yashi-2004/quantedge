"""
QuantEdge — Page 04: Econometric Research & Statistical Finance
Rigorous econometric modeling: ADF/KPSS, auto-ARIMA, GARCH(1,1)/EGARCH, VAR, Kalman Filter, Diebold-Mariano.
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.components.ui import THEME, apply_terminal_theme, create_plotly_figure, render_metric_card
from src.data.ingestion import DEMO_UNIVERSE, MarketDataIngestion
from src.econometrics.arima import ARIMAModeler
from src.econometrics.garch import GARCHModeler
from src.econometrics.kalman import KalmanFilterEngine
from src.econometrics.stationarity import StationarityAnalyzer
from src.econometrics.tests import StatisticalTestLab
from src.econometrics.var import VARModeler

st.set_page_config(page_title="Econometrics | QuantEdge", page_icon="📐", layout="wide")
apply_terminal_theme()

if "ingestion" not in st.session_state:
    st.session_state.ingestion = MarketDataIngestion()
ingestion: MarketDataIngestion = st.session_state.ingestion

st.markdown("## 📐 Econometric Research & Statistical Finance Lab")
st.caption("Time-series stationarity, auto-ARIMA, GARCH volatility clustering, Vector Autoregression, and Kalman filtering.")

c1, c2 = st.columns([2, 1])
with c1:
    all_demo = DEMO_UNIVERSE["US"] + DEMO_UNIVERSE["INDIA"]
    ticker = st.selectbox("Primary Time Series", options=all_demo, index=0)
with c2:
    series_type = st.radio("Modeling Target", options=["Daily Log Returns", "Raw Price Level"], index=0, horizontal=True)

res = ingestion.fetch_market_data(ticker)
df = res.df

if series_type == "Daily Log Returns":
    target_series = np.log(df["Close"] / df["Close"].shift(1)).dropna().rename(f"{ticker}_LogReturn")
else:
    target_series = df["Close"].rename(f"{ticker}_Close")

tab_stationarity, tab_arima, tab_garch, tab_var, tab_kalman, tab_tests = st.tabs([
    "1. Stationarity & ACF",
    "2. Auto-ARIMA",
    "3. GARCH Volatility",
    "4. Vector Autoregression (VAR)",
    "5. Kalman Filter",
    "6. Hypothesis Testing Lab",
])

# 1. Stationarity
with tab_stationarity:
    st.markdown("### Dual ADF and KPSS Stationarity Testing")
    stat_report = StationarityAnalyzer.test_stationarity(target_series)

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        render_metric_card("ADF Statistic", f"{stat_report.adf_statistic:.4f}", f"p = {stat_report.adf_pvalue:.4f}")
    with sc2:
        render_metric_card("KPSS Statistic", f"{stat_report.kpss_statistic:.4f}", f"p = {stat_report.kpss_pvalue:.4f}")
    with sc3:
        render_metric_card("Stationary?", "YES (Mean-Reverting)" if stat_report.is_stationary else "NO (Unit Root Present)", is_positive=stat_report.is_stationary)

    st.info(f"**Interpretation:** {stat_report.interpretation}")

    # ACF / PACF Plots
    acf_vals, pacf_vals, confint = StationarityAnalyzer.compute_acf_pacf(target_series, nlags=20)
    fig_acf = create_plotly_figure()
    fig_acf.add_trace(go.Bar(x=list(range(len(acf_vals))), y=acf_vals, name="Autocorrelation (ACF)", marker_color=THEME["accent_blue"]))
    fig_acf.add_hline(y=1.96 / np.sqrt(len(target_series)), line_dash="dash", line_color=THEME["accent_red"], annotation_text="+95% CI")
    fig_acf.add_hline(y=-1.96 / np.sqrt(len(target_series)), line_dash="dash", line_color=THEME["accent_red"], annotation_text="-95% CI")
    fig_acf.update_layout(xaxis_title="Lag", yaxis_title="ACF", height=350)
    st.plotly_chart(fig_acf, use_container_width=True)

# 2. Auto-ARIMA
with tab_arima:
    st.markdown("### Automated ARIMA Order Selection & Forecasting")
    c_ar1, c_ar2 = st.columns(2)
    with c_ar1:
        fc_steps = st.slider("Forecast Steps Ahead (Trading Days)", min_value=5, max_value=30, value=10)
    with c_ar2:
        d_val = st.selectbox("Differencing Order d", options=[0, 1], index=0 if stat_report.is_stationary else 1)

    if st.button("🚀 Fit Optimal ARIMA Model"):
        with st.spinner("Executing AIC grid search..."):
            arima_mod = ARIMAModeler()
            arima_res = arima_mod.fit_auto_arima(target_series, max_p=3, d=d_val, max_q=3, forecast_steps=fc_steps)

        ac1, ac2, ac3 = st.columns(3)
        with ac1:
            render_metric_card("Selected Order", f"ARIMA{arima_res.order}")
        with ac2:
            render_metric_card("Akaike Info Criterion (AIC)", f"{arima_res.aic:,.1f}")
        with ac3:
            render_metric_card("Residuals White Noise?", "YES" if arima_res.is_white_noise else "NO (Autocorr Left)", is_positive=arima_res.is_white_noise)

        fig_arima = create_plotly_figure()
        # Plot last 60 actual observations
        tail_series = target_series.tail(60)
        fig_arima.add_trace(go.Scatter(x=tail_series.index, y=tail_series, name="Recent Observed", line=dict(color="#FFFFFF", width=1.5)))
        fig_arima.add_trace(go.Scatter(x=arima_res.forecast_mean.index, y=arima_res.forecast_mean, name="ARIMA Forecast Mean", line=dict(color=THEME["accent_teal"], width=2.5)))
        fig_arima.add_trace(go.Scatter(
            x=list(arima_res.forecast_ci_upper.index) + list(arima_res.forecast_ci_lower.index)[::-1],
            y=list(arima_res.forecast_ci_upper) + list(arima_res.forecast_ci_lower)[::-1],
            fill="toself",
            fillcolor="rgba(0, 212, 178, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="95% Confidence Band",
        ))
        fig_arima.update_layout(yaxis_title="Target Value", height=420)
        st.plotly_chart(fig_arima, use_container_width=True)

# 3. GARCH
with tab_garch:
    st.markdown("### GARCH Family Volatility Modeling")
    st.caption("Models volatility clustering, conditional heteroskedasticity, and shock persistence.")
    
    garch_type = st.selectbox("Model Architecture", options=["GARCH", "EGARCH", "GJR-GARCH"], index=0)
    ret_for_garch = np.log(df["Close"] / df["Close"].shift(1)).dropna()

    if st.button("⚡ Estimate Conditional Volatility"):
        with st.spinner("Optimizing quasi-maximum likelihood..."):
            garch_mod = GARCHModeler()
            g_res = garch_mod.fit_garch(ret_for_garch, model_type=garch_type, forecast_horizon=15)

        gc1, gc2, gc3, gc4 = st.columns(4)
        with gc1:
            render_metric_card("Persistence (α + β)", f"{g_res.persistence:.4f}", "High Memory" if g_res.persistence > 0.9 else "Low Memory")
        with gc2:
            render_metric_card("Shock Half-Life", f"{g_res.half_life_days} Days")
        with gc3:
            render_metric_card("Unconditional Annual Vol", f"{g_res.unconditional_vol_annual*100:.2f}%")
        with gc4:
            render_metric_card("ARCH-LM Test p-val", f"{g_res.arch_lm_pvalue:.4f}", "Clustering Captured" if g_res.arch_lm_pvalue > 0.05 else "Residual Arch Left")

        fig_garch = create_plotly_figure()
        fig_garch.add_trace(go.Scatter(x=g_res.conditional_volatility.index, y=g_res.conditional_volatility * 100, name="Conditional Volatility (σ_t %)", line=dict(color=THEME["accent_amber"], width=1.5)))
        fig_garch.update_layout(yaxis_title="Annualized Volatility (%)", height=400)
        st.plotly_chart(fig_garch, use_container_width=True)

# 4. VAR
with tab_var:
    st.markdown("### Vector Autoregression (VAR)")
    st.caption("Studies dynamic interdependencies and impulse responses across multiple series.")
    
    spy_res = ingestion.fetch_market_data("SPY")
    gld_res = ingestion.fetch_market_data("GLD")
    combined_var_df = pd.DataFrame({
        f"{ticker}_Ret": ret_for_garch,
        "SPY_Ret": spy_res.df["Close"].pct_change(),
        "GLD_Ret": gld_res.df["Close"].pct_change(),
    }).dropna()

    if st.button("Fit Multivariate VAR"):
        var_mod = VARModeler()
        var_res = var_mod.fit_var(combined_var_df, max_lags=4, forecast_steps=10)

        vc1, vc2, vc3 = st.columns(3)
        with vc1:
            render_metric_card("Selected Lag Order (AIC)", f"Lag {var_res.selected_lag}")
        with vc2:
            render_metric_card("Characteristic Root Modulus", f"{var_res.max_root_modulus:.4f}")
        with vc3:
            render_metric_card("VAR System Stable?", "STABLE" if var_res.is_stable else "UNSTABLE", is_positive=var_res.is_stable)

        st.markdown("#### Forecast Error Variance Decomposition (FEVD) (%)")
        st.dataframe(var_res.fevd_df, use_container_width=True)

# 5. Kalman Filter
with tab_kalman:
    st.markdown("### Kalman Filtering: Latent Fundamental Price Trend")
    st.caption("Extracts true unobserved state from noisy market microstructure observations.")
    
    obs_noise = st.slider("Observation Microstructure Noise Variance R", min_value=0.1, max_value=5.0, value=1.0)
    proc_noise = st.slider("Process True Trend Variance Q", min_value=0.001, max_value=0.1, value=0.01, step=0.005)

    kalman = KalmanFilterEngine()
    k_res = kalman.filter_price_trend(df["Close"], observation_noise_var=obs_noise, process_noise_var=proc_noise)

    fig_kalman = create_plotly_figure()
    fig_kalman.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Observed Market Price", line=dict(color="#718096", width=1)))
    fig_kalman.add_trace(go.Scatter(x=k_res.filtered_trend.index, y=k_res.filtered_trend, name="Kalman Filtered Trend (x̂_t)", line=dict(color=THEME["accent_teal"], width=2)))
    fig_kalman.update_layout(yaxis_title=f"Price ({res.currency})", height=420)
    st.plotly_chart(fig_kalman, use_container_width=True)

# 6. Hypothesis Testing Lab
with tab_tests:
    st.markdown("### Statistical Hypothesis Verification Lab")
    ret_sample = np.log(df["Close"] / df["Close"].shift(1)).dropna()

    tc1, tc2 = st.columns(2)
    with tc1:
        jb_res = StatisticalTestLab.test_normality(ret_sample)
        st.markdown(f"**{jb_res.test_name}**")
        st.caption(f"Decision: `{jb_res.decision}` (Statistic={jb_res.statistic}, p={jb_res.p_value})")
        st.write(jb_res.interpretation)

    with tc2:
        lb_res = StatisticalTestLab.test_autocorrelation(ret_sample, lags=10)
        st.markdown(f"**{lb_res.test_name}**")
        st.caption(f"Decision: `{lb_res.decision}` (Statistic={lb_res.statistic}, p={lb_res.p_value})")
        st.write(lb_res.interpretation)

    st.divider()
    st.markdown("#### Stationary Block Bootstrap: 95% Confidence Interval for Sharpe Ratio")
    sharpe_orig, sh_low, sh_high = StatisticalTestLab.stationary_bootstrap_sharpe(ret_sample, num_bootstraps=500)
    bc1, bc2, bc3 = st.columns(3)
    with bc1:
        render_metric_card("Empirical Sample Sharpe", f"{sharpe_orig:.2f}")
    with bc2:
        render_metric_card("95% CI Lower Bound", f"{sh_low:.2f}")
    with bc3:
        render_metric_card("95% CI Upper Bound", f"{sh_high:.2f}")
