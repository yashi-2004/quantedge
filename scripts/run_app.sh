#!/usr/bin/env bash
# QuantEdge Launch Script
# Configures environment for local CPU execution and starts the Streamlit terminal

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

export MPLCONFIGDIR="/tmp/matplotlib"
export HF_HOME="/tmp/huggingface"
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
mkdir -p /tmp/matplotlib /tmp/huggingface

echo "=========================================================="
echo "⚡ Starting QuantEdge Institutional Quantitative Workstation"
echo "=========================================================="
echo "Platform: Local CPU Mode (Zero External Paid APIs)"
echo "Directory: $PROJECT_DIR"
echo "Listening on: http://localhost:8501"
echo "=========================================================="

./.venv/bin/streamlit run app/main.py --server.headless=true --server.port=8501
