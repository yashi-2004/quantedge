"""
QuantEdge — Page 06: Deep Learning for Financial Time Series
PyTorch CPU-optimized neural network architectures: LSTM, GRU, TCN, and Time-Series Transformer.
Sliding window temporal splits, early stopping, and loss curves.
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
from src.features.pipeline import FeatureEngine
from src.models.classical import ClassicalMLSuite
from src.models.deep_models import DeepLearningLab

st.set_page_config(page_title="Deep Learning | QuantEdge", page_icon="🧠", layout="wide")
apply_terminal_theme()

if "ingestion" not in st.session_state:
    st.session_state.ingestion = MarketDataIngestion()
ingestion: MarketDataIngestion = st.session_state.ingestion

st.markdown("## 🧠 Deep Learning for Financial Time Series")
st.caption("PyTorch sequential neural networks: Recurrent (LSTM/GRU), Convolutional (TCN), and Attention (Transformer).")

c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    all_demo = DEMO_UNIVERSE["US"] + DEMO_UNIVERSE["INDIA"]
    ticker = st.selectbox("Underlying Asset", options=all_demo, index=0)
with c2:
    arch = st.selectbox("Neural Architecture", options=["LSTM", "GRU", "TCN", "Transformer"], index=0)
with c3:
    seq_len = st.slider("Lookback Window (Sequence Length)", min_value=5, max_value=30, value=15)

col_ep, col_lr, col_bs = st.columns(3)
with col_ep:
    epochs = st.slider("Training Epochs", min_value=10, max_value=50, value=25, step=5)
with col_lr:
    lr = st.select_slider("Learning Rate", options=[0.0005, 0.001, 0.003, 0.005, 0.01], value=0.003)
with col_bs:
    batch_size = st.selectbox("Batch Size (CPU)", options=[16, 32, 64], index=1)

res = ingestion.fetch_market_data(ticker)
engine = FeatureEngine(include_macro=True)
fmat = engine.build_feature_matrix(res.df)

ml_suite = ClassicalMLSuite()
X_tr, _, X_te, y_tr, _, y_te = ml_suite.chronological_split(fmat.X, fmat.y_return, train_ratio=0.75, val_ratio=0.0)

st.markdown(f"**Sequence Data:** {len(X_tr)} train rows | {len(X_te)} test rows | Input dimensions = {X_tr.shape[1]} features.")

if st.button(f"🚀 Train {arch} Network"):
    with st.spinner(f"Training {arch} on CPU (Adam optimizer + Early Stopping)..."):
        dl_lab = DeepLearningLab(seq_len=seq_len, batch_size=batch_size, epochs=epochs, lr=lr)
        dl_res = dl_lab.train_model(arch, X_tr, y_tr, X_te, y_te)

    # Top Metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_card("Test RMSE", f"{dl_res.test_metrics['RMSE']:.5f}")
    with m2:
        render_metric_card("Directional Accuracy", f"{dl_res.test_metrics['Directional_Accuracy']:.1f}%", is_positive=dl_res.test_metrics['Directional_Accuracy'] > 50.0)
    with m3:
        render_metric_card("Information Coeff (IC)", f"{dl_res.test_metrics['IC']:+.4f}", is_positive=dl_res.test_metrics['IC'] > 0)
    with m4:
        render_metric_card("Epochs Executed", f"{dl_res.total_epochs} / {epochs}", "Early Stopping Active" if dl_res.total_epochs < epochs else "Full Run")

    # Plots: Loss Curves and Out-of-Sample Predictions
    col_loss, col_preds = st.columns(2)

    with col_loss:
        st.markdown("### Training & Validation Loss Curves (MSE)")
        fig_loss = create_plotly_figure()
        ep_axis = list(range(1, len(dl_res.train_losses) + 1))
        fig_loss.add_trace(go.Scatter(x=ep_axis, y=dl_res.train_losses, name="Train Loss", line=dict(color=THEME["accent_blue"], width=2)))
        fig_loss.add_trace(go.Scatter(x=ep_axis, y=dl_res.val_losses, name="Val Loss", line=dict(color=THEME["accent_red"], width=2, dash="dash")))
        fig_loss.update_layout(xaxis_title="Epoch", yaxis_title="Loss", height=380)
        st.plotly_chart(fig_loss, use_container_width=True)

    with col_preds:
        st.markdown(f"### {arch} Test Set Forecasts vs Actuals")
        fig_dl_pred = create_plotly_figure()
        fig_dl_pred.add_trace(go.Scatter(x=dl_res.actuals_test.index, y=dl_res.actuals_test, name="Actual Return", line=dict(color="#718096", width=1.5)))
        fig_dl_pred.add_trace(go.Scatter(x=dl_res.predictions_test.index, y=dl_res.predictions_test, name=f"{arch} Prediction", line=dict(color=THEME["accent_teal"], width=2)))
        fig_dl_pred.update_layout(yaxis_title="Return", height=380)
        st.plotly_chart(fig_dl_pred, use_container_width=True)

    st.info(
        "💡 **Hypothesis H1 Note:** Financial time series frequently display low signal-to-noise ratios. "
        "Deep architectures risk learning transient sample noise unless heavily regularized or coupled with regime filtering."
    )
