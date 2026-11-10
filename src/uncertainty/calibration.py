"""
QuantEdge Uncertainty Quantification & Conformal Prediction Engine
Implements:
1. Quantile Regression for asymmetric upper/lower predictive bands (10th, 50th, 90th percentiles)
2. Conformal Prediction for distribution-free finite-sample coverage guarantees
3. Residual Distribution & Calibration Diagnostics
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler


@dataclass
class UncertaintyForecastResult:
    median_forecast: pd.Series
    lower_bound: pd.Series  # e.g., 10th percentile
    upper_bound: pd.Series  # e.g., 90th percentile
    nominal_coverage: float  # e.g., 0.80
    empirical_coverage: float  # Actual percentage of test points within interval
    mean_interval_width: float
    is_well_calibrated: bool


class UncertaintyForecaster:
    """
    Quantifies predictive epistemic and aleatoric uncertainty in financial time series.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def quantile_regression_bands(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        lower_alpha: float = 0.10,
        upper_alpha: float = 0.90,
    ) -> UncertaintyForecastResult:
        """
        Fits 3 Gradient Boosting models minimizing pinball loss for lower, median, and upper quantiles.
        """
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_te = scaler.transform(X_test)

        # Lower bound
        mod_lower = GradientBoostingRegressor(
            loss="quantile", alpha=lower_alpha, n_estimators=60, max_depth=3, random_state=self.random_state
        )
        mod_lower.fit(X_tr, y_train)
        pred_lower = mod_lower.predict(X_te)

        # Median
        mod_median = GradientBoostingRegressor(
            loss="quantile", alpha=0.50, n_estimators=60, max_depth=3, random_state=self.random_state
        )
        mod_median.fit(X_tr, y_train)
        pred_median = mod_median.predict(X_te)

        # Upper bound
        mod_upper = GradientBoostingRegressor(
            loss="quantile", alpha=upper_alpha, n_estimators=60, max_depth=3, random_state=self.random_state
        )
        mod_upper.fit(X_tr, y_train)
        pred_upper = mod_upper.predict(X_te)

        # Ensure monotonic bounds (lower <= median <= upper)
        pred_lower = np.minimum(pred_lower, pred_median)
        pred_upper = np.maximum(pred_upper, pred_median)

        y_actual = y_test.values
        in_bounds = (y_actual >= pred_lower) & (y_actual <= pred_upper)
        empirical_cov = float(np.mean(in_bounds) * 100.0)
        nominal_cov = float((upper_alpha - lower_alpha) * 100.0)
        mean_width = float(np.mean(pred_upper - pred_lower))

        well_calibrated = abs(empirical_cov - nominal_cov) <= 10.0

        return UncertaintyForecastResult(
            median_forecast=pd.Series(pred_median, index=y_test.index, name="Median_Forecast"),
            lower_bound=pd.Series(pred_lower, index=y_test.index, name=f"Lower_{int(lower_alpha*100)}%"),
            upper_bound=pd.Series(pred_upper, index=y_test.index, name=f"Upper_{int(upper_alpha*100)}%"),
            nominal_coverage=nominal_cov,
            empirical_coverage=round(empirical_cov, 2),
            mean_interval_width=round(mean_width, 6),
            is_well_calibrated=well_calibrated,
        )

    def conformal_prediction_intervals(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_calib: pd.DataFrame,
        y_calib: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        alpha: float = 0.10,  # 90% confidence
    ) -> UncertaintyForecastResult:
        """
        Split conformal prediction for rigorous finite-sample coverage guarantees.
        """
        from sklearn.linear_model import Ridge

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_ca = scaler.transform(X_calib)
        X_te = scaler.transform(X_test)

        base_model = Ridge(alpha=1.0)
        base_model.fit(X_tr, y_train)

        # Non-conformity scores on calibration set: s_i = |y_i - y_hat_i|
        calib_preds = base_model.predict(X_ca)
        scores = np.abs(y_calib.values - calib_preds)

        # Conformal quantile: q_hat = ceil((n+1)(1-alpha)) / n quantile
        n = len(scores)
        q_level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
        q_hat = float(np.quantile(scores, q_level))

        # Test point predictions
        test_preds = base_model.predict(X_te)
        lower_bound = test_preds - q_hat
        upper_bound = test_preds + q_hat

        in_bounds = (y_test.values >= lower_bound) & (y_test.values <= upper_bound)
        empirical_cov = float(np.mean(in_bounds) * 100.0)
        nominal_cov = float((1.0 - alpha) * 100.0)

        return UncertaintyForecastResult(
            median_forecast=pd.Series(test_preds, index=y_test.index, name="Conformal_Median"),
            lower_bound=pd.Series(lower_bound, index=y_test.index, name="Conformal_Lower"),
            upper_bound=pd.Series(upper_bound, index=y_test.index, name="Conformal_Upper"),
            nominal_coverage=nominal_cov,
            empirical_coverage=round(empirical_cov, 2),
            mean_interval_width=round(2.0 * q_hat, 6),
            is_well_calibrated=abs(empirical_cov - nominal_cov) <= 10.0,
        )
