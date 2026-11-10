"""
QuantEdge ARIMA and SARIMAX Modeling Engine
Provides automated order selection, multi-step out-of-sample forecasting with confidence intervals,
and rigorous residual diagnostics.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.arima.model import ARIMA


@dataclass
class ARIMAResult:
    order: Tuple[int, int, int]
    aic: float
    bic: float
    log_likelihood: float
    in_sample_fitted: pd.Series
    forecast_mean: pd.Series
    forecast_ci_lower: pd.Series
    forecast_ci_upper: pd.Series
    residuals: pd.Series
    ljung_box_pvalue: float
    jarque_bera_pvalue: float
    is_white_noise: bool
    summary_text: str


class ARIMAModeler:
    """
    Fits ARIMA(p,d,q) processes on financial returns or price differences.
    """

    def fit_auto_arima(
        self,
        series: pd.Series,
        max_p: int = 3,
        d: int = 1,
        max_q: int = 3,
        forecast_steps: int = 10,
        alpha: float = 0.05,
    ) -> ARIMAResult:
        clean = series.dropna().astype(float)
        best_aic = float("inf")
        best_order = (1, d, 1)
        best_model = None

        # Grid search over (p, d, q)
        for p in range(0, max_p + 1):
            for q in range(0, max_q + 1):
                try:
                    mod = ARIMA(clean, order=(p, d, q)).fit()
                    if mod.aic < best_aic:
                        best_aic = mod.aic
                        best_order = (p, d, q)
                        best_model = mod
                except Exception:
                    continue

        if best_model is None:
            # Fallback to simple random walk / AR(1)
            best_order = (1, d, 0)
            best_model = ARIMA(clean, order=best_order).fit()

        # In-sample fitted values and residuals
        fitted_vals = best_model.fittedvalues
        residuals = best_model.resid

        # Multi-step out-of-sample forecast
        fc_res = best_model.get_forecast(steps=forecast_steps)
        fc_mean = fc_res.predicted_mean
        ci = fc_res.conf_int(alpha=alpha)

        # Dates for forecast
        if isinstance(clean.index, pd.DatetimeIndex):
            freq = pd.infer_freq(clean.index) or "B"
            fc_dates = pd.date_range(start=clean.index[-1], periods=forecast_steps + 1, freq=freq)[1:]
            fc_mean.index = fc_dates
            ci.index = fc_dates

        ci_lower = ci.iloc[:, 0]
        ci_upper = ci.iloc[:, 1]

        # Residual Diagnostics
        lb_df = acorr_ljungbox(residuals.dropna(), lags=[10], return_df=True)
        lb_pval = float(lb_df["lb_pvalue"].iloc[0])
        jb_stat, jb_pval, _, _ = jarque_bera(residuals.dropna())

        # Residuals are white noise if Ljung-Box fails to reject H0 (p >= 0.05)
        is_white_noise = lb_pval >= 0.05

        return ARIMAResult(
            order=best_order,
            aic=round(float(best_model.aic), 2),
            bic=round(float(best_model.bic), 2),
            log_likelihood=round(float(best_model.llf), 2),
            in_sample_fitted=fitted_vals,
            forecast_mean=fc_mean,
            forecast_ci_lower=ci_lower,
            forecast_ci_upper=ci_upper,
            residuals=residuals,
            ljung_box_pvalue=round(lb_pval, 4),
            jarque_bera_pvalue=round(float(jb_pval), 4),
            is_white_noise=is_white_noise,
            summary_text=str(best_model.summary()),
        )
