"""
QuantEdge — Page 07: Financial Sentiment Intelligence
Natural Language Processing sentiment scoring using FinBERT / Loughran-McDonald lexicon.
Temporal aggregation and return correlation analysis.
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
from src.sentiment.sentiment_engine import SentimentAnalyzer

st.set_page_config(page_title="Financial Sentiment | QuantEdge", page_icon="📰", layout="wide")
apply_terminal_theme()

if "ingestion" not in st.session_state:
    st.session_state.ingestion = MarketDataIngestion()
ingestion: MarketDataIngestion = st.session_state.ingestion

st.markdown("## 📰 Financial Sentiment Intelligence")
st.caption("Quantitative text processing for market headlines using domain-specific sentiment modeling.")

analyzer = SentimentAnalyzer(use_transformer=False)

tab_live, tab_curated, tab_impact = st.tabs([
    "🔍 Real-Time Headline Scoring",
    "📚 Curated Feed & Sentiment Series",
    "📈 Sentiment vs Price Impact",
])

with tab_live:
    st.markdown("### Interactive Single Headline Scorer")
    sample_default = "NVIDIA beats quarterly revenue expectations on surge in enterprise datacenter demand"
    user_headline = st.text_input("Enter Financial Headline or News Snippet", value=sample_default)

    if user_headline:
        score = analyzer.analyze_headline(user_headline)

        s1, s2, s3, s4 = st.columns(4)
        with s1:
            render_metric_card("Sentiment Classification", score.sentiment_label.upper())
        with s2:
            render_metric_card("Compound Score", f"{score.compound_score:+.3f}", is_positive=score.compound_score > 0)
        with s3:
            render_metric_card("Positive Probability", f"{score.positive_prob*100:.1f}%")
        with s4:
            render_metric_card("Negative Probability", f"{score.negative_prob*100:.1f}%")

        # Probability Bar Chart
        fig_prob = go.Figure(go.Bar(
            x=["Positive", "Neutral", "Negative"],
            y=[score.positive_prob, score.neutral_prob, score.negative_prob],
            marker_color=[THEME["accent_green"], THEME["text_muted"], THEME["accent_red"]],
        ))
        fig_prob.update_layout(
            template="plotly_dark",
            paper_bgcolor=THEME["bg_card"],
            plot_bgcolor=THEME["bg_secondary"],
            yaxis_title="Probability",
            height=300,
            margin=dict(l=40, r=40, t=20, b=40),
        )
        st.plotly_chart(fig_prob, use_container_width=True)

with tab_curated:
    st.markdown("### Historical Sentiment Aggregation Feed")
    ticker_choice = st.selectbox("Select Asset Feed", options=["SPY", "AAPL", "NVDA", "RELIANCE.NS"], index=0)
    
    news_df = analyzer.get_curated_sample_news(ticker=ticker_choice, n_days=90)
    summary = analyzer.analyze_batch(news_df)

    sm1, sm2, sm3, sm4 = st.columns(4)
    with sm1:
        render_metric_card("Total Headlines", f"{summary.total_headlines}")
    with sm2:
        render_metric_card("Bullish Headlines", f"{summary.positive_pct:.1f}%", is_positive=True)
    with sm3:
        render_metric_card("Bearish Headlines", f"{summary.negative_pct:.1f}%", is_positive=False)
    with sm4:
        render_metric_card("Neutral Headlines", f"{summary.neutral_pct:.1f}%")

    st.markdown("#### Daily Aggregated Sentiment Index")
    fig_sent = create_plotly_figure()
    fig_sent.add_trace(go.Bar(
        x=summary.sentiment_df.index,
        y=summary.sentiment_df["Daily_Sentiment_Mean"],
        name="Daily Mean Sentiment",
        marker_color=np.where(summary.sentiment_df["Daily_Sentiment_Mean"] >= 0, THEME["accent_green"], THEME["accent_red"]),
    ))
    fig_sent.update_layout(yaxis_title="Compound Sentiment (-1.0 to +1.0)", height=350)
    st.plotly_chart(fig_sent, use_container_width=True)

    st.dataframe(news_df.tail(20), use_container_width=True)

with tab_impact:
    st.markdown("### Sentiment vs Forward Returns Alignment")
    st.caption("Evaluates whether extreme positive or negative sentiment leads to price continuation or mean reversion.")

    price_res = ingestion.fetch_market_data(ticker_choice)
    aligned_df = pd.concat([price_res.df["Close"].pct_change().rename("Return"), summary.sentiment_df["Daily_Sentiment_Mean"]], axis=1).dropna()

    if len(aligned_df) > 10:
        fig_scatter = px.scatter(
            aligned_df,
            x="Daily_Sentiment_Mean",
            y="Return",
            trendline="ols",
            template="plotly_dark",
            labels={"Daily_Sentiment_Mean": "Daily Sentiment Index", "Return": "Asset Return"},
        )
        fig_scatter.update_layout(
            paper_bgcolor=THEME["bg_card"],
            plot_bgcolor=THEME["bg_secondary"],
            height=400,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        corr_val = aligned_df.corr().iloc[0, 1]
        st.info(f"Empirical correlation between Daily Sentiment and Asset Return: `{corr_val:+.4f}`")
