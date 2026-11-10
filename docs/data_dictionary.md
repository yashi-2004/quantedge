# QuantEdge Data Dictionary & Feature Taxonomy

## 1. Primary Market Data Fields (OHLCV)

| Column | Type | Unit | Description |
| :--- | :--- | :--- | :--- |
| `Date` | Datetime (Index) | ISO 8601 | Trading session date (harmonized calendar). |
| `Open` | Float64 | Currency (USD / INR) | Price of the first transaction of the trading session. |
| `High` | Float64 | Currency (USD / INR) | Maximum recorded transaction price of the session. |
| `Low` | Float64 | Currency (USD / INR) | Minimum recorded transaction price of the session. |
| `Close` | Float64 | Currency (USD / INR) | Unadjusted or adjusted closing auction price. |
| `Adj Close` | Float64 | Currency (USD / INR) | Backward-adjusted closing price for corporate actions and dividends. |
| `Volume` | Int64 | Shares / Units | Aggregate number of shares or units transacted during the session. |

---

## 2. Engineered Return & Statistical Features

| Feature Identifier | Mathematical Definition | Financial Interpretation |
| :--- | :--- | :--- |
| `Return_Simple` | $R_t = \frac{P_t - P_{t-1}}{P_{t-1}}$ | Percentage capital change between consecutive closes. |
| `Return_Log` | $r_t = \ln(P_t / P_{t-1})$ | Continuously compounded logarithmic return. |
| `Overnight_Gap` | $\frac{\text{Open}_t - \text{Close}_{t-1}}{\text{Close}_{t-1}}$ | Return realized outside normal regular trading hours. |
| `Intraday_Range` | $\frac{\text{High}_t - \text{Low}_t}{\text{Open}_t}$ | Relative dispersion of market sentiment during the session. |
| `Lag_Ret_k` | $R_{t-k}, \quad k \in \{1, 2, 3, 5, 10, 21\}$ | Historical return realized $k$ periods prior. |
| `Roll_Mean_w` | $\frac{1}{w}\sum_{i=0}^{w-1} R_{t-i}$ | Rolling expected return over window $w \in \{21, 63\}$. |
| `Roll_Std_w` | $\sqrt{\frac{1}{w-1}\sum (R_{t-i} - \bar{R})^2}$ | Rolling close-to-close volatility estimator. |
| `Roll_Skew_w` | $\frac{1}{w}\sum \left(\frac{R_{t-i} - \bar{R}}{s}\right)^3$ | Third standardized moment (asymmetry of return shocks). |
| `Roll_Kurt_w` | $\frac{1}{w}\sum \left(\frac{R_{t-i} - \bar{R}}{s}\right)^4 - 3$ | Fourth standardized moment (excess tail heaviness). |
| `Autocorr_Lag1_63`| $\text{Corr}(R_t, R_{t-1})_{63}$ | Rolling short-term persistence vs mean-reversion tendencies. |
| `Mean_Rev_Z_20` | $\frac{P_t - \text{SMA}_{20}(P)}{\sigma_{20}(P)}$ | Standardized distance from the 20-day trend. |

---

## 3. Advanced Range-Based Volatility Estimators

| Identifier | Mathematical Formula | Statistical Properties |
| :--- | :--- | :--- |
| `Vol_Realized_21` | $\sqrt{252} \cdot s_{\text{close}}$ | Baseline close-to-close sample standard deviation. |
| `Vol_Parkinson_21`| $\sqrt{\frac{252}{4 \ln 2} \frac{1}{N}\sum \ln(H_t/L_t)^2}$ | High-Low range estimator; ~5x more statistically efficient than CC. |
| `Vol_Garman_Klass_21` | $\sqrt{\frac{252}{N}\sum \left[0.5\ln\left(\frac{H_t}{L_t}\right)^2 - (2\ln 2 - 1)\ln\left(\frac{C_t}{O_t}\right)^2\right]}$ | Incorporates Open, High, Low, Close; up to 8x efficiency. |
| `Vol_Rogers_Satchell_21` | $\sqrt{\frac{252}{N}\sum \left[\ln\frac{H_t}{C_t}\ln\frac{H_t}{O_t} + \ln\frac{L_t}{C_t}\ln\frac{L_t}{O_t}\right]}$ | Robust to non-zero price drift during trading hours. |
| `Vol_Yang_Zhang_21` | $\sqrt{\sigma_o^2 + k\sigma_c^2 + (1-k)\sigma_{RS}^2} \cdot \sqrt{252}$ | Minimum variance unbiased estimator robust to overnight gaps. |
| `Vol_Of_Vol_21` | $\text{Std}(\sigma_{t-21:t})$ | Volatility of volatility; proxy for market turbulence risk. |

---

## 4. Macroeconomic & Regime Indicators

| Indicator | Source / Model | Description |
| :--- | :--- | :--- |
| `Macro_Yield_10Y` | Public FRED Benchmark / Proxy | Yield to maturity of the benchmark 10-Year US Treasury Note. |
| `Macro_Yield_2Y` | Public FRED Benchmark / Proxy | Yield to maturity of the 2-Year Treasury (monetary policy proxy). |
| `Macro_Yield_Curve_Slope`| $Y_{10} - Y_{2}$ | Yield curve slope. Negative values denote inverted yield curve. |
| `Macro_Risk_Free_Rate`| 3-Month T-Bill / SOFR Proxy | Annualized risk-free benchmark for Sharpe and Sortino computation. |
| `Macro_VIX_Proxy` | CBOE VIX Index / Jump Model | Expected 30-day implied equity market volatility. |

---

## 5. NLP Sentiment Metrics

| Variable | Bounds | Description |
| :--- | :--- | :--- |
| `Compound_Score` | $[-1.0, +1.0]$ | Aggregate net sentiment ($P(\text{Positive}) - P(\text{Negative})$). |
| `Positive_Prob` | $[0.0, 1.0]$ | FinBERT / lexicon probability of bullish financial context. |
| `Negative_Prob` | $[0.0, 1.0]$ | FinBERT / lexicon probability of bearish financial context. |
| `Neutral_Prob` | $[0.0, 1.0]$ | Probability that headline conveys factual, neutral market reporting. |
