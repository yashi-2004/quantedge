"""
QuantEdge Statistical Testing & Hypothesis Verification Engine
Implements:
- Jarque-Bera Test (Normality)
- Ljung-Box Test (Serial Autocorrelation)
- ARCH-LM Test (Engle 1982 Heteroskedasticity)
- Diebold-Mariano Test (Comparing Competing Forecasts)
- Stationary Bootstrap for Sharpe Ratios and Return Distributions
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera


@dataclass
class TestResult:
    test_name: str
    statistic: float
    p_value: float
    null_hypothesis: str
    decision: str  # Reject H0 or Fail to Reject H0
    interpretation: str


@dataclass
class DieboldMarianoResult:
    dm_statistic: float
    p_value: float
    better_model: str
    loss_metric: str
    interpretation: str


class StatisticalTestLab:
    """
    Hypothesis testing and diagnostic suite for financial time series research.
    """

    @staticmethod
    def test_normality(series: pd.Series) -> TestResult:
        clean = series.dropna().values
        jb_stat, jb_pval, skew, kurt = jarque_bera(clean)
        reject = jb_pval < 0.05
        decision = "Reject H0 (Non-Normal)" if reject else "Fail to Reject H0 (Normal)"
        interp = (
            f"Series shows significant deviation from Gaussian distribution (Skewness={skew:.2f}, Kurtosis={kurt:.2f}). "
            "Heavy tails or asymmetry present; parametric normal models (e.g. standard Black-Scholes) will underestimate tail risk."
            if reject
            else "No significant evidence against normality at alpha=0.05."
        )
        return TestResult(
            test_name="Jarque-Bera Normality Test",
            statistic=round(float(jb_stat), 4),
            p_value=round(float(jb_pval), 6),
            null_hypothesis="Returns are identically and normally distributed (Skew=0, Excess Kurtosis=0).",
            decision=decision,
            interpretation=interp,
        )

    @staticmethod
    def test_autocorrelation(series: pd.Series, lags: int = 10) -> TestResult:
        clean = series.dropna()
        res = acorr_ljungbox(clean, lags=[lags], return_df=True)
        lb_stat = float(res["lb_stat"].iloc[0])
        lb_pval = float(res["lb_pvalue"].iloc[0])
        reject = lb_pval < 0.05
        decision = "Reject H0 (Autocorrelation Present)" if reject else "Fail to Reject H0 (White Noise)"
        interp = (
            f"Significant serial correlation detected up to lag {lags} (p={lb_pval:.4f}). "
            "Past returns or residuals contain predictive linear structure, violating weak-form Random Walk."
            if reject
            else f"No significant autocorrelation detected up to lag {lags}. Sequence behaves like white noise."
        )
        return TestResult(
            test_name=f"Ljung-Box Test (lags={lags})",
            statistic=round(lb_stat, 4),
            p_value=round(lb_pval, 6),
            null_hypothesis="Series has zero serial autocorrelation up to lag k.",
            decision=decision,
            interpretation=interp,
        )

    @staticmethod
    def test_arch_effect(residuals: pd.Series, lags: int = 5) -> TestResult:
        clean = residuals.dropna().values
        lm_stat, p_val, _, _ = het_arch(clean, maxlag=lags)
        reject = p_val < 0.05
        decision = "Reject H0 (Volatility Clustering Present)" if reject else "Fail to Reject H0 (Homoskedastic)"
        interp = (
            "Statistically significant ARCH effects (time-varying variance) detected. "
            "Volatility clustering is active; GARCH-type models are mathematically justified."
            if reject
            else "No significant autoregressive conditional heteroskedasticity found."
        )
        return TestResult(
            test_name=f"Engle ARCH-LM Test (lags={lags})",
            statistic=round(float(lm_stat), 4),
            p_value=round(float(p_val), 6),
            null_hypothesis="Residual variance is homoskedastic (no autoregressive conditional variance).",
            decision=decision,
            interpretation=interp,
        )

    @staticmethod
    def diebold_mariano_test(
        actual: np.ndarray,
        pred_model_1: np.ndarray,
        pred_model_2: np.ndarray,
        criterion: str = "MSE",  # 'MSE' or 'MAE'
        h: int = 1,
    ) -> DieboldMarianoResult:
        """
        Diebold-Mariano test for equal predictive accuracy of two competing forecasts:
        H0: E[d_t] = 0 (Both models have identical forecast accuracy)
        H1: Forecast accuracy differs significantly
        """
        actual = np.asarray(actual)
        p1 = np.asarray(pred_model_1)
        p2 = np.asarray(pred_model_2)

        e1 = actual - p1
        e2 = actual - p2

        if criterion.upper() == "MAE":
            d = np.abs(e1) - np.abs(e2)
        else:
            d = (e1**2) - (e2**2)

        mean_d = np.mean(d)
        n = len(d)

        # Autocovariance adjustment (Newey-West variance of differential loss for h-step ahead forecasts)
        gamma0 = np.var(d)
        gamma_sum = 0.0
        for i in range(1, h):
            cov = np.sum((d[i:] - mean_d) * (d[:-i] - mean_d)) / n
            gamma_sum += 2.0 * cov

        var_d = (gamma0 + gamma_sum) / n
        if var_d <= 1e-12:
            dm_stat = 0.0
            p_val = 1.0
        else:
            dm_stat = mean_d / np.sqrt(var_d)
            p_val = 2.0 * (1.0 - stats.norm.cdf(np.abs(dm_stat)))

        if p_val < 0.05:
            better = "Model 1" if mean_d < 0 else "Model 2"
            interp = f"Statistically significant difference in predictive accuracy at 5% level (p={p_val:.4f}). {better} demonstrates superior out-of-sample accuracy."
        else:
            better = "Neither (Statistically Equivalent)"
            interp = f"No statistically significant difference in forecast accuracy (p={p_val:.4f}). Models perform equivalently on this sample."

        return DieboldMarianoResult(
            dm_statistic=round(float(dm_stat), 4),
            p_value=round(float(p_val), 4),
            better_model=better,
            loss_metric=criterion.upper(),
            interpretation=interp,
        )

    @staticmethod
    def stationary_bootstrap_sharpe(
        returns: pd.Series,
        num_bootstraps: int = 1000,
        expected_block_len: int = 10,
        alpha: float = 0.05,
    ) -> Tuple[float, float, float]:
        """
        Politis & Romano (1994) Stationary Bootstrap to construct confidence intervals
        for the annualized Sharpe ratio without assuming independent Gaussian returns.
        """
        r = returns.dropna().values
        n = len(r)
        if n < 20:
            return 0.0, 0.0, 0.0

        p = 1.0 / expected_block_len
        sharpes = np.zeros(num_bootstraps)

        for b in range(num_bootstraps):
            indices = np.zeros(n, dtype=int)
            idx = np.random.randint(0, n)
            for t in range(n):
                if np.random.rand() < p:
                    idx = np.random.randint(0, n)
                else:
                    idx = (idx + 1) % n
                indices[t] = idx

            sample = r[indices]
            s_std = np.std(sample)
            sharpes[b] = (np.mean(sample) / (s_std + 1e-12)) * np.sqrt(252)

        original_sharpe = float((np.mean(r) / (np.std(r) + 1e-12)) * np.sqrt(252))
        lower = float(np.percentile(sharpes, 100 * (alpha / 2.0)))
        upper = float(np.percentile(sharpes, 100 * (1.0 - alpha / 2.0)))

        return round(original_sharpe, 2), round(lower, 2), round(upper, 2)
