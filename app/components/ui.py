"""
QuantEdge UI Design System & Terminal Styling
Provides consistent dark navy/charcoal styling, cards, metrics, and Plotly templates.
"""

from typing import Dict, List, Optional
import plotly.graph_objects as go
import streamlit as st

THEME = {
    "bg_primary": "#0B0E14",
    "bg_secondary": "#151922",
    "bg_card": "#1A202C",
    "border": "#2D3748",
    "accent_teal": "#00D4B2",
    "accent_blue": "#38BDF8",
    "accent_green": "#34D399",
    "accent_red": "#F87171",
    "accent_amber": "#FBBF24",
    "text_primary": "#F7FAFC",
    "text_secondary": "#A0AEC0",
    "text_muted": "#718096",
}


def apply_terminal_theme():
    """Injects custom dark quantitative terminal CSS styling."""
    st.markdown(
        f"""
        <style>
            /* Main container */
            .stApp {{
                background-color: {THEME['bg_primary']};
                color: {THEME['text_primary']};
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            }}
            
            /* Sidebar styling */
            section[data-testid="stSidebar"] {{
                background-color: {THEME['bg_secondary']} !important;
                border-right: 1px solid {THEME['border']};
            }}
            
            /* Card Containers */
            .quant-card {{
                background-color: {THEME['bg_card']};
                border: 1px solid {THEME['border']};
                border-radius: 8px;
                padding: 18px;
                margin-bottom: 16px;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
            }}
            
            .quant-card:hover {{
                border-color: {THEME['accent_teal']}44;
            }}
            
            /* Custom Metric Cards */
            .metric-box {{
                background: linear-gradient(135deg, {THEME['bg_secondary']} 0%, {THEME['bg_card']} 100%);
                border: 1px solid {THEME['border']};
                border-radius: 6px;
                padding: 12px 16px;
                margin-bottom: 10px;
            }}
            
            .metric-title {{
                font-size: 0.8rem;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                color: {THEME['text_secondary']};
                margin-bottom: 4px;
            }}
            
            .metric-value {{
                font-size: 1.5rem;
                font-weight: 700;
                color: {THEME['text_primary']};
                font-family: "SF Mono", "Fira Code", Menlo, monospace;
            }}
            
            .metric-delta-pos {{
                color: {THEME['accent_green']};
                font-size: 0.85rem;
                font-weight: 600;
            }}
            
            .metric-delta-neg {{
                color: {THEME['accent_red']};
                font-size: 0.85rem;
                font-weight: 600;
            }}

            /* Badges */
            .badge-live {{
                background-color: #065F46;
                color: #A7F3D0;
                padding: 2px 8px;
                border-radius: 4px;
                font-size: 0.75rem;
                font-weight: 600;
            }}
            
            .badge-sample {{
                background-color: #92400E;
                color: #FDE68A;
                padding: 2px 8px;
                border-radius: 4px;
                font-size: 0.75rem;
                font-weight: 600;
            }}
            
            .badge-cached {{
                background-color: #1E3A8A;
                color: #BFDBFE;
                padding: 2px 8px;
                border-radius: 4px;
                font-size: 0.75rem;
                font-weight: 600;
            }}

            /* Headers */
            h1, h2, h3, h4 {{
                color: {THEME['text_primary']} !important;
                font-weight: 600;
                letter-spacing: -0.02em;
            }}
            
            /* Dataframe / Tables */
            .stDataFrame {{
                border-radius: 6px;
                border: 1px solid {THEME['border']};
            }}
        </style>
    """,
        unsafe_allow_html=True,
    )


def create_plotly_figure() -> go.Figure:
    """Returns a Plotly Figure pre-configured with the QuantEdge terminal theme."""
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=THEME["bg_card"],
        plot_bgcolor=THEME["bg_secondary"],
        font=dict(color=THEME["text_primary"], family="sans-serif", size=12),
        margin=dict(l=40, r=40, t=50, b=40),
        xaxis=dict(
            gridcolor=THEME["border"],
            showgrid=True,
            zerolinecolor=THEME["border"],
        ),
        yaxis=dict(
            gridcolor=THEME["border"],
            showgrid=True,
            zerolinecolor=THEME["border"],
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor=THEME["border"],
            font=dict(size=11),
        ),
    )
    return fig


def render_metric_card(title: str, value: str, delta: Optional[str] = None, is_positive: Optional[bool] = None):
    delta_html = ""
    if delta:
        delta_class = "metric-delta-pos" if is_positive else ("metric-delta-neg" if is_positive is False else "")
        delta_html = f"<div class='{delta_class}'>{delta}</div>"
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            {delta_html}
        </div>
    """,
        unsafe_allow_html=True,
    )
