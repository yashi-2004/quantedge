"""
QuantEdge — Page 05: Machine Learning Forecasting Laboratory
Classical ML, Optuna Bayesian hyperparameter optimization, and benchmark baseline comparisons.
Strictly enforced chronological time-series evaluation.
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
from src.models.baseline import FinancialBaselines
from src.models.classical import ClassicalMLSuite
from src.models.tuning import HyperparameterTuner

st.set_page_config(page_title="Machine Learning | QuantEdge", page_icon="🤖", layout="wide")
apply_terminal_theme()

if "ingestion" not in st.session_state:
    st.session_state.ingestion = MarketDataIngestion()
ingestion: MarketDataIngestion = st.session_state.ingestion

st.markdown("## 🤖 Classical Machine Learning Forecasting Laboratory")
st.caption("Regularized linear regressions, ensemble decision trees, Optuna hyperparameter optimization, and baselines.")

# Model & Data Configuration
c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    all_demo = DEMO_UNIVERSE["US"] + DEMO_UNIVERSE["INDIA"]
    ticker = st.selectbox("Target Instrument", options=all_demo, index=0)
with c2:
    model_name = st.selectbox("Algorithm", options=ClassicalMLSuite.SUPPORTED_REGRESSION, index=1)
with c3:
    task_target = st.selectbox("Prediction Target", options=["Next-Day Return (R_{t+1})", "Forward 5-Day Volatility"], index=0)

res = ingestion.fetch_market_data(ticker)
engine = FeatureEngine(include_macro=True)
fmat = engine.build_feature_matrix(res.df)

target_series = fmat.y_return if "Return" in task_target else fmat.y_volatility

# Chronological Split
ml_suite = ClassicalMLSuite()
X_tr, X_va, X_te, y_tr, y_va, y_te = ml_suite.chronological_split(fmat.X, target_series)

st.markdown(f"**Dataset Partitioning:** Training ({len(X_tr)} sessions: {X_tr.index[0].date()} to {X_tr.index[-1].date()}) | Validation ({len(X_va)} sessions) | Out-of-Sample Test ({len(X_te)} sessions: {X_te.index[0].date()} to {X_te.index[-1].date()})")

col_opt, col_train = st.columns([1, 1])
with col_opt:
    run_optuna = st.checkbox("Tune Hyperparameters with Optuna (Bayesian CV)", value=False)

if st.button("🚀 Train & Evaluate Model on Test Set"):
    best_params = {}
    if run_optuna:
        with st.spinner("Running Optuna Bayesian hyperparameter search..."):
            tuner = HyperparameterTuner(n_splits=3, n_trials=10)
            tune_res = tuner.tune_xgboost(X_tr, y_tr)
            best_params = tune_res.best_params
            st.success(f"Optuna optimal parameters found: {best_params}")

    with st.spinner("Training model on chronological train partition..."):
        eval_res = ml_suite.train_regression_model(
            model_name, X_tr, y_tr, X_te, y_te, params=best_params, scale_features=True
        )

    # Baselines comparison
    baselines = FinancialBaselines()
    base_metrics = baselines.compute_all_baselines(y_tr, y_te)

    # Top metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_card("Test RMSE", f"{eval_res.out_of_sample_metrics['RMSE']:.5f}")
    with m2:
        render_metric_card("Directional Accuracy", f"{eval_res.out_of_sample_metrics['Directional_Accuracy']:.1f}%", is_positive=eval_res.out_of_sample_metrics['Directional_Accuracy'] > 50.0)
    with m3:
        render_metric_card("Information Coeff (IC)", f"{eval_res.out_of_sample_metrics['IC']:+.4f}", is_positive=eval_res.out_of_sample_metrics['IC'] > 0)
    with m4:
        render_metric_card("Out-of-Sample R²", f"{eval_res.out_of_sample_metrics['R2']:.4f}", is_positive=eval_res.out_of_sample_metrics['R2'] > 0)

    # Model vs Baselines Comparison Table
    st.markdown("### Out-of-Sample Performance vs Baselines")
    comp_rows = [
        {
            "Model / Benchmark": f"Selected Model ({model_name})",
            "RMSE": eval_res.out_of_sample_metrics["RMSE"],
            "MAE": eval_res.out_of_sample_metrics["MAE"],
            "Directional Accuracy (%)": eval_res.out_of_sample_metrics["Directional_Accuracy"],
            "Rank IC": eval_res.out_of_sample_metrics["IC"],
            "R-Squared": eval_res.out_of_sample_metrics["R2"],
        },
        {
            "Model / Benchmark": "Zero-Return Baseline (Martingale)",
            "RMSE": base_metrics["Zero_Return"].rmse,
            "MAE": base_metrics["Zero_Return"].mae,
            "Directional Accuracy (%)": base_metrics["Zero_Return"].directional_accuracy,
            "Rank IC": base_metrics["Zero_Return"].information_coefficient,
            "R-Squared": base_metrics["Zero_Return"].r_squared,
        },
        {
            "Model / Benchmark": "Historical Mean Baseline",
            "RMSE": base_metrics["Historical_Mean"].rmse,
            "MAE": base_metrics["Historical_Mean"].mae,
            "Directional Accuracy (%)": base_metrics["Historical_Mean"].directional_accuracy,
            "Rank IC": base_metrics["Historical_Mean"].information_coefficient,
            "R-Squared": base_metrics["Historical_Mean"].r_squared,
        },
    ]
    st.dataframe(pd.DataFrame(comp_rows).set_index("Model / Benchmark"), use_container_width=True)

    # Visualizations
    col_pred_plot, col_feat_imp = st.columns(2)

    with col_pred_plot:
        st.markdown("### Out-of-Sample Predictions vs Actuals")
        fig_pred = create_plotly_figure()
        fig_pred.add_trace(go.Scatter(x=y_te.index, y=y_te, name="Actual Outcome", line=dict(color="#718096", width=1.5)))
        fig_pred.add_trace(go.Scatter(x=eval_res.predictions_test.index, y=eval_res.predictions_test, name="Model Forecast", line=dict(color=THEME["accent_teal"], width=2)))
        fig_pred.update_layout(yaxis_title="Target Value", height=380)
        st.plotly_chart(fig_pred, use_container_width=True)

    with col_feat_imp:
        st.markdown("### Top Feature Importances / Coefficients")
        top_imp = eval_res.feature_importances.head(12)
        fig_imp = px.bar(
            x=top_imp.values,
            y=top_imp.index,
            orientation="h",
            template="plotly_dark",
            color=top_imp.values,
            color_continuous_scale="Teal",
        )
        fig_imp.update_layout(
            paper_bgcolor=THEME["bg_card"],
            plot_bgcolor=THEME["bg_secondary"],
            xaxis_title="Importance Score",
            yaxis_title="Feature",
            height=380,
        )
        st.plotly_chart(fig_imp, use_container_width=True)
