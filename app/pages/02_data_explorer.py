"""
QuantEdge — Page 02: Data Explorer & Quality Assurance
Ingests, validates, remediates, and audits raw financial time series with strict provenance tracking.
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import pandas as pd
import streamlit as st

from app.components.ui import THEME, apply_terminal_theme, render_metric_card
from src.data.ingestion import DEMO_UNIVERSE, MarketDataIngestion

st.set_page_config(page_title="Data Explorer | QuantEdge", page_icon="🔍", layout="wide")
apply_terminal_theme()

if "ingestion" not in st.session_state:
    st.session_state.ingestion = MarketDataIngestion()
ingestion: MarketDataIngestion = st.session_state.ingestion

st.markdown("## 🔍 Market Data Explorer & Data Integrity Lab")
st.caption("Inspect raw OHLCV series, examine data validation reports, audit provenance, or upload custom datasets.")

tab_inspect, tab_upload, tab_registry = st.tabs([
    "📋 Inspect Ingested Ticker",
    "📤 Upload Custom CSV",
    "🗄️ SQLite Database Registry",
])

with tab_inspect:
    c1, c2 = st.columns([2, 1])
    with c1:
        all_demo = DEMO_UNIVERSE["US"] + DEMO_UNIVERSE["INDIA"]
        ticker = st.selectbox("Select Security to Audit", options=all_demo, index=0)
    with c2:
        use_adj = st.checkbox("View Adjusted Close", value=True)

    with st.spinner("Auditing dataset integrity..."):
        res = ingestion.fetch_market_data(ticker, use_adjusted=use_adj)
        val = res.validation
        df = res.df

    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_card("Data Quality Score", f"{val.quality_score} / 100", is_positive=val.quality_score >= 80)
    with m2:
        render_metric_card("Total Sessions", f"{val.total_records:,}")
    with m3:
        render_metric_card("Price Inversions", f"{val.inversion_count}", is_positive=val.inversion_count == 0)
    with m4:
        render_metric_card("Data Provenance", res.provenance)

    # Validation Details Card
    st.markdown("### Data Quality Diagnostics & Anomaly Report")
    with st.container():
        st.markdown(f"""
        <div class="quant-card">
            <h4 style="margin-top:0;">Validation Summary for <code>{ticker}</code></h4>
            <ul>
                <li><strong>Dataset Status:</strong> {'✅ VALID' if val.is_valid else '⚠️ DEFECTS DETECTED'}</li>
                <li><strong>Date Coverage:</strong> {val.start_date} to {val.end_date}</li>
                <li><strong>Missing Price Records:</strong> {val.missing_count} ({val.missing_pct:.2f}%)</li>
                <li><strong>Duplicate Timestamps:</strong> {val.duplicate_count}</li>
                <li><strong>Detected Statistical Outliers:</strong> {val.outlier_count}</li>
                <li><strong>Zero/Negative Volume Sessions:</strong> {val.zero_volume_count}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    if val.issues:
        st.warning("⚠️ **Detected Issues:**\n- " + "\n- ".join(val.issues))
    if val.remediation_applied:
        st.info("🔧 **Applied Quantitative Remediations:**\n- " + "\n- ".join(val.remediation_applied))

    # Raw Data Table
    st.markdown("### Raw Historical Price Data")
    st.dataframe(df.tail(100), use_container_width=True)

    # CSV Download Button
    csv_bytes = df.to_csv().encode("utf-8")
    st.download_button(
        label=f"⬇️ Export {ticker} Cleaned OHLCV (CSV)",
        data=csv_bytes,
        file_name=f"{ticker}_cleaned_ohlcv.csv",
        mime="text/csv",
    )

with tab_upload:
    st.markdown("### Upload Custom Asset CSV")
    st.caption("Upload historical daily prices. Automatically maps date, open, high, low, close, volume.")
    
    uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])
    custom_name = st.text_input("Asset Identifier / Ticker Name", value="CUSTOM_ASSET")

    if uploaded_file is not None:
        try:
            content = uploaded_file.read()
            up_res = ingestion.parse_uploaded_csv(content, ticker=custom_name)
            st.success(f"Successfully ingested and cleaned {len(up_res.df)} sessions for {custom_name}!")
            
            u_col1, u_col2 = st.columns(2)
            with u_col1:
                render_metric_card("Upload Quality Score", f"{up_res.validation.quality_score} / 100")
            with u_col2:
                render_metric_card("Total Uploaded Rows", f"{len(up_res.df)}")

            st.dataframe(up_res.df.head(20), use_container_width=True)
        except Exception as e:
            st.error(f"Error parsing uploaded CSV: {e}")

with tab_registry:
    st.markdown("### Local Storage Registry")
    st.caption("Cached datasets indexed in local SQLite store (`quantedge_data.db`).")
    reg_df = ingestion.storage.get_registered_assets()
    if not reg_df.empty:
        st.dataframe(reg_df, use_container_width=True)
    else:
        st.info("No assets currently registered in SQLite database.")
