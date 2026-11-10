"""
QuantEdge Vector Autoregression (VAR) Modeling Engine
Performs multivariate time series modeling across interrelated assets and macroeconomic factors:
- Lag order selection (AIC / BIC / FPE)
- Characteristic root stability analysis
- Impulse Response Functions (IRF)
- Forecast Error Variance Decomposition (FEVD)
- Dynamic multi-step forecasting
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR


@dataclass
class VARResult:
    selected_lag: int
    is_stable: bool
    max_root_modulus: float
    variables: List[str]
    forecast_df: pd.DataFrame
    fevd_df: pd.DataFrame  # Forecast Error Variance Decomposition at step H
    irf_periods: int
    irf_matrices: Dict[str, pd.DataFrame]  # Impulse response trajectories
    aic: float
    bic: float
    summary_text: str


class VARModeler:
    """
    Fits Vector Autoregressive models to study inter-asset spillover and macro feedback.
    """

    def fit_var(
        self,
        df: pd.DataFrame,
        max_lags: int = 5,
        forecast_steps: int = 10,
        irf_periods: int = 10,
    ) -> VARResult:
        clean_df = df.dropna().astype(float)
        if clean_df.shape[1] < 2:
            raise ValueError("VAR requires at least 2 time series columns.")
        if len(clean_df) < (max_lags * clean_df.shape[1] + 20):
            raise ValueError("Insufficient time series observations for stable VAR estimation.")

        model = VAR(clean_df)
        lag_order_res = model.select_order(maxlags=max_lags)
        best_lag = int(lag_order_res.aic)
        best_lag = max(1, min(best_lag, max_lags))

        fitted = model.fit(best_lag)

        # Stability Check: Check eigenvalues of companion matrix
        roots = fitted.roots
        max_root = float(np.max(np.abs(roots))) if len(roots) > 0 else 0.0
        is_stable = max_root < 1.0  # Stable if strictly inside unit circle

        # Multi-step forecast
        fc_vals = fitted.forecast(clean_df.values[-best_lag:], steps=forecast_steps)
        fc_dates = None
        if isinstance(clean_df.index, pd.DatetimeIndex):
            freq = pd.infer_freq(clean_df.index) or "B"
            fc_dates = pd.date_range(start=clean_df.index[-1], periods=forecast_steps + 1, freq=freq)[1:]

        fc_df = pd.DataFrame(
            fc_vals,
            index=fc_dates if fc_dates is not None else [f"T+{i+1}" for i in range(forecast_steps)],
            columns=clean_df.columns,
        )

        # Impulse Response Functions (IRF) with Cholesky orthogonalization
        irf = fitted.irf(irf_periods)
        irf_dict: Dict[str, pd.DataFrame] = {}
        vars_list = list(clean_df.columns)
        for i, var_impulse in enumerate(vars_list):
            for j, var_response in enumerate(vars_list):
                key = f"Shock_{var_impulse} -> {var_response}"
                irf_series = irf.orth_irfs[:, j, i]
                irf_dict[key] = pd.DataFrame(
                    {"Period": list(range(irf_periods + 1)), "Response": irf_series}
                )

        # Forecast Error Variance Decomposition (FEVD) at horizon
        fevd = fitted.fevd(irf_periods)
        fevd_arr = fevd.decomp[-1]  # At final period
        fevd_df = pd.DataFrame(fevd_arr * 100.0, index=vars_list, columns=[f"Shock_{v}" for v in vars_list])

        return VARResult(
            selected_lag=best_lag,
            is_stable=is_stable,
            max_root_modulus=round(max_root, 4),
            variables=vars_list,
            forecast_df=fc_df,
            fevd_df=fevd_df.round(2),
            irf_periods=irf_periods,
            irf_matrices=irf_dict,
            aic=round(float(fitted.aic), 2),
            bic=round(float(fitted.bic), 2),
            summary_text=str(fitted.summary()),
        )
