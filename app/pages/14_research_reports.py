"""
QuantEdge — Page 14: Quantitative Research Paper & Automated Reporting Engine
Compiles empirical evidence across econometric, ML, DL, and portfolio experiments.
Tests core hypotheses H1 to H5 and exports publication-ready research reports.
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

from datetime import datetime
import json
import pandas as pd
import streamlit as st

from app.components.ui import THEME, apply_terminal_theme, render_metric_card
from src.data.ingestion import DEMO_UNIVERSE
from src.research.reporting import ResearchReportGenerator

st.set_page_config(page_title="Research Reports | QuantEdge", page_icon="📄", layout="wide")
apply_terminal_theme()

st.markdown("## 📄 Quantitative Research Paper & Reporting Engine")
st.caption("Empirical evaluation of core research hypotheses (H1–H5), statistical significance, and automated report generation.")

report_gen = ResearchReportGenerator()
hypotheses = report_gen.evaluate_core_hypotheses()

tab_hypotheses, tab_paper, tab_export = st.tabs([
    "🔬 Empirical Hypothesis Evaluation (H1–H5)",
    "📖 Full Academic Research Paper",
    "💾 Export Center (Markdown / HTML / JSON)",
])

with tab_hypotheses:
    st.markdown("### Formal Research Hypotheses & Findings")
    st.caption("Pre-registered experimental hypotheses evaluated across the demonstration asset universe.")

    for h in hypotheses:
        status_color = THEME["accent_green"] if h.empirical_status == "SUPPORTED" else THEME["accent_amber"]
        with st.container():
            st.markdown(
                f"""
                <div class="quant-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <h4 style="margin:0; color:#38BDF8;">{h.hypothesis_id}: {h.statement}</h4>
                        <span style="background-color:rgba(52,211,153,0.2); color:{status_color}; padding:3px 10px; border-radius:4px; font-weight:700; font-size:0.8rem;">
                            {h.empirical_status}
                        </span>
                    </div>
                    <div style="margin-top:8px; font-size:0.9rem; color:#A0AEC0;">
                        <strong>Key Test Statistic:</strong> <code>{h.test_statistic}</code>
                    </div>
                    <p style="margin-top:8px; font-size:0.95rem; color:#F7FAFC;">
                        {h.evidence_summary}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

with tab_paper:
    st.markdown("### Full Working Paper Preview")
    
    universe = DEMO_UNIVERSE["US"][:5]
    md_content = report_gen.generate_markdown_report(
        universe=universe,
        start_date="2021-01-01",
        end_date=datetime.now().strftime("%Y-%m-%d"),
    )
    st.markdown(md_content)

with tab_export:
    st.markdown("### Export Research Artifacts")
    st.caption("Download generated reports for academic and institutional research presentations.")

    ec1, ec2, ec3 = st.columns(3)

    # 1. Markdown Export
    with ec1:
        st.download_button(
            label="⬇️ Export Markdown Paper (.md)",
            data=md_content.encode("utf-8"),
            file_name=f"quantedge_research_paper_{datetime.now().strftime('%Y%m%d')}.md",
            mime="text/markdown",
        )

    # 2. HTML Export
    with ec2:
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>QuantEdge Research Report</title>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0B0E14; color: #F7FAFC; padding: 40px; line-height: 1.6; max-width: 900px; margin: auto; }}
                h1, h2, h3, h4 {{ color: #00D4B2; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; border: 1px solid #2D3748; }}
                th, td {{ border: 1px solid #2D3748; padding: 10px; text-align: left; }}
                th {{ background: #151922; color: #38BDF8; }}
                code {{ background: #1A202C; padding: 2px 6px; border-radius: 4px; color: #FBBF24; }}
                hr {{ border: 0; border-top: 1px solid #2D3748; margin: 30px 0; }}
            </style>
        </head>
        <body>
            <pre style="white-space: pre-wrap; font-family: inherit;">{md_content}</pre>
        </body>
        </html>
        """
        st.download_button(
            label="⬇️ Export Standalone HTML (.html)",
            data=html_content.encode("utf-8"),
            file_name=f"quantedge_research_paper_{datetime.now().strftime('%Y%m%d')}.html",
            mime="text/html",
        )

    # 3. JSON Structured Findings
    with ec3:
        findings_json = json.dumps(
            [
                {
                    "hypothesis": h.hypothesis_id,
                    "statement": h.statement,
                    "status": h.empirical_status,
                    "statistic": h.test_statistic,
                    "evidence": h.evidence_summary,
                }
                for h in hypotheses
            ],
            indent=2,
        )
        st.download_button(
            label="⬇️ Export Hypotheses Data (.json)",
            data=findings_json.encode("utf-8"),
            file_name="quantedge_hypotheses_evaluation.json",
            mime="application/json",
        )
