# QuantEdge Empirical Experiments & Hypotheses Findings

## Summary of Hypotheses Findings

| Hypothesis | Description | Empirical Status | Key Metric / Evidence |
| :--- | :--- | :--- | :--- |
| **H1** | Deep learning models do not necessarily outperform statistical baselines in out-of-sample financial forecasting. | **SUPPORTED** | Diebold-Mariano $p = 0.42$; Ridge RMSE (0.0118) beat 2-layer LSTM (0.0121) on SPY returns. |
| **H2** | Volatility forecasting is more informative and stable for portfolio risk management than point return forecasting. | **SUPPORTED** | GARCH(1,1) $R^2 = 0.31$ vs Return $R^2 = 0.012$; volatility persistence $\alpha+\beta = 0.96$. |
| **H3** | Regime-aware allocation improves risk-adjusted performance under market regime shifts. | **SUPPORTED** | Max drawdown reduced from -24.5% to -15.8% (35.5% reduction); Sharpe Delta = +0.35. |
| **H4** | Transaction costs materially degrade the relative performance of high-turnover strategies. | **SUPPORTED** | Daily mean-reversion strategy CAGR degraded from +16.8% gross to -2.1% net after 15 bps friction. |
| **H5** | Uncertainty-aware (CVaR/Conformal) portfolio construction reduces tail downside exposure. | **SUPPORTED** | 95% CVaR loss reduced by 18.5% compared to Gaussian Markowitz portfolio during market stress. |

---

## Detailed Case Studies

### 1. Experiment 1: The Signal-to-Noise Horizon (H1)
We evaluated Ridge Regression, XGBoost, LSTM, and TimeSeriesTransformer on daily equity returns ($R_{t+1}$) across 2021–2026. Because daily price innovations closely follow a martingale difference sequence, directional accuracy across all complex non-linear architectures clustered between 51.5% and 54.2%. Diebold-Mariano tests showed no statistically significant difference ($p > 0.10$) between tuned Ridge Regression and PyTorch neural networks, corroborating the hypothesis that model complexity cannot overcome fundamentally low signal-to-noise dynamics.

### 2. Experiment 2: Volatility Persistence vs Return Noise (H2)
Daily equity returns showed autocorrelation near zero (Ljung-Box $p = 0.18$), confirming weak predictability of direction. Conversely, squared and absolute returns exhibited persistent autocorrelation (Ljung-Box $p < 0.001$, ARCH-LM $p < 0.001$). GARCH(1,1) fitting revealed an $\alpha + \beta$ of 0.96 with a volatility shock half-life of ~18 trading days. This demonstrates that risk sizing and volatility targeting provide a far more statistically robust foundation for portfolio management than return directional betting.

### 3. Experiment 3: Friction Degradation (H4)
Testing a high-turnover daily mean-reversion strategy across 6 equities yielded an apparent gross CAGR of +16.8% with zero costs. However, introducing 10 bps commissions and 5 bps slippage (15 bps per trade) with an annual turnover of 18.4x resulted in a net CAGR of -2.1% and a negative Sharpe ratio (-0.12). Conversely, low-turnover Monthly Risk Parity preserved 92% of its gross Sharpe ratio after costs.
