# QuantEdge System Limitations & Risk Disclosures

## 1. Methodological Limitations

1. **Survivorship Bias in Static Universes**: The default demonstration universe consists of currently prominent securities (e.g., SPY, AAPL, NVDA, RELIANCE.NS). In historical backtests extending 5–10 years into the past, evaluating only current large-cap survivors can overestimate historical returns. Users are encouraged to test custom CSV datasets including delisted or restructured instruments.
2. **Non-Stationarity & Structural Breaks**: Macroeconomic regimes, central bank interest rate policies, and geopolitical shocks fundamentally alter data-generating processes. A model trained on a low-interest-rate bull market (2012–2020) may perform poorly in an inflationary rate-hike environment (2022–2024).
3. **Capacity & Market Impact Non-Linearities**: QuantEdge models execution friction using linear proportional slippage (default 5 bps). In reality, institutional trade execution suffers from square-root law market impact and non-linear order book exhaustion that varies with market depth and time of day.
4. **Sentiment Data Granularity**: The sentiment intelligence engine analyzes headlines at daily aggregated timestamps. It does not model high-frequency millisecond microstructure news feeds or SEC filing 10-K NLP parsing.

---

## 2. Hardware and Compute Constraints

1. **Local CPU Execution**: Deep learning models (LSTM, TCN, Transformers) and Optuna studies are designed with lightweight parameter configurations suited for quad-core Intel Core i5 and Apple Silicon CPUs.
2. **Batch Sequences**: Sequence lengths are constrained to $T \le 30$ days and batch sizes to 16–64 to guarantee smooth interactive performance within Streamlit without memory exhaustion.

---

## 3. Regulatory & Educational Disclaimer

**NOT FINANCIAL ADVICE.**  
QuantEdge is an academic and educational quantitative research workstation built strictly for statistical research, portfolio mathematics demonstration, and econometric modeling. The software does not execute live trades, does not connect to brokerages, and makes no representations regarding the future profitability of any systematic strategy or model.
