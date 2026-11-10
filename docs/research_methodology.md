# QuantEdge Research Methodology & Experimental Framework

## 1. Research Motivation & Core Working Title

**A Comparative Study of Econometric, Machine Learning and Deep Learning Approaches for Financial Forecasting and Risk-Aware Portfolio Construction**

Modern quantitative finance is situated at the intersection of classical econometrics (efficient markets, random walks, GARCH volatility clustering) and modern financial machine learning. A persistent issue in retail quantitative tools is the uncritical application of complex neural networks to raw price levels, ignoring non-stationarity, market microstructure, transaction friction, and low signal-to-noise dynamics.

QuantEdge establishes a rigorous empirical benchmark to answer:
1. When (and why) do regularized linear models beat multi-layer deep neural networks?
2. Why is volatility modeling mathematically tractable while return prediction approaches white noise?
3. How severely do realistic transaction costs degrade high-turnover algorithmic alpha?
4. Can Bayesian view-combining (Black-Litterman) and tail-risk optimization (CVaR) protect capital better than unconstrained Markowitz mean-variance optimization?

---

## 2. Pre-Registered Research Hypotheses

| ID | Hypothesis Statement | Theoretical Grounding | Empirical Evaluation Tool |
| :--- | :--- | :--- | :--- |
| **H1** | Deep learning models do not necessarily outperform classical statistical baselines in out-of-sample financial forecasting. | Non-stationarity, regime shifts, and low signal-to-noise ratio cause high-capacity neural networks to overfit noise. | Diebold-Mariano test, test RMSE, out-of-sample directional accuracy, Information Coefficient (IC). |
| **H2** | Volatility forecasting is more informative and stable for portfolio risk management than point return forecasting. | Volatility exhibits long memory, clustering ($\alpha+\beta \approx 1$), and non-zero autocorrelation of squared returns. | ARCH-LM test, GARCH(1,1) persistence parameter, half-life of shocks, variance ratio test. |
| **H3** | Regime-aware allocation improves risk-adjusted performance under market regime shifts. | Asset correlation spikes to 1.0 during liquidity crises; shifting defensive mitigates tail drawdown. | 3-State Gaussian Hidden Markov Model, transition matrix, regime-conditional Sharpe ratio delta. |
| **H4** | Transaction costs materially degrade the relative performance of high-turnover strategies. | Proportional execution friction (10-15 bps per round trip) compounds over high rebalancing frequencies, turning gross alpha negative. | Backtest simulation with vs without friction, turnover tracking, net CAGR degradation. |
| **H5** | Uncertainty-aware (CVaR/Conformal) portfolio construction reduces tail downside exposure. | Financial return innovations exhibit heavy tails (Student-t with df ≈ 4–5), invalidating Gaussian variance assumptions. | Historical & Parametric CVaR, Monte Carlo severe bear quantiles, historical stress test drawdowns. |

---

## 3. Strict Anti-Leakage Protocol

To ensure publication-grade validity, the following causal constraints are enforced across the platform:

1. **Temporal Train/Validation/Test Splits**: Time-series cross-validation (`TimeSeriesSplit`) or chronological slicing (70% train / 15% validation / 15% test). Never shuffle timestamps.
2. **Standardization Fit on Train Only**: All scalers (`StandardScaler`, `RobustScaler`) fit parameters exclusively on the chronological training partition.
3. **Point-in-Time Forward Targets**: Features at index $t$ utilize price and volume information strictly up to session $t$. The target $R_{t+1}$ represents return realized over $t \to t+1$.
4. **Execution Lag**: Signals generated at session close $t$ are executed at session $t+1$ (or $t+2$), capturing realistic operational and order-routing latencies.
