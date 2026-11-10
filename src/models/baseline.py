"""
QuantEdge Benchmark Baselines Engine
Establishes rigorous financial baselines that machine learning models must beat:
- Zero-Return Forecast (Market Efficiency null hypothesis)
- Historical Mean Return
- Historical Volatility Baseline
- Naive Persistence Forecast
"""

from dataclasses import dataclass
from typing import Dict
import numpy as np
import pandas as pd


@dataclass
class BaselineMetrics:
    name: str
    rmse: float
    mae: float
    directional_accuracy: float
    information_coefficient: float  # Rank correlation
    r_squared: float


class FinancialBaselines:
    """
    Standard zero-cost benchmarks for financial returns and volatility forecasting.
    """

    @staticmethod
    def evaluate_forecast(actual: np.ndarray, predicted: np.ndarray, name: str) -> BaselineMetrics:
        actual = np.asarray(actual)
        predicted = np.asarray(predicted)
        errors = actual - predicted
        rmse = float(np.sqrt(np.mean(errors**2)))
        mae = float(np.mean(np.abs(errors)))

        # Directional accuracy: sign agreement
        actual_dir = actual > 0
        pred_dir = predicted > 0
        hit_ratio = float(np.mean(actual_dir == pred_dir))

        # Information Coefficient (Spearman Rank Correlation)
        from scipy.stats import spearmanr

        ic_val, _ = spearmanr(predicted, actual)
        ic = float(ic_val) if not np.isnan(ic_val) else 0.0

        # R-squared out-of-sample
        ss_res = np.sum(errors**2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = float(1.0 - (ss_res / (ss_tot + 1e-12)))

        return BaselineMetrics(
            name=name,
            rmse=round(rmse, 6),
            mae=round(mae, 6),
            directional_accuracy=round(hit_ratio * 100.0, 2),
            information_coefficient=round(ic, 4),
            r_squared=round(r2, 4),
        )

    def compute_all_baselines(self, y_train: pd.Series, y_test: pd.Series) -> Dict[str, BaselineMetrics]:
        actual = y_test.values
        n = len(actual)

        # 1. Zero-Return Baseline (E[R] = 0)
        pred_zero = np.zeros(n)
        res_zero = self.evaluate_forecast(actual, pred_zero, "Zero-Return Baseline")

        # 2. Historical Mean Baseline (E[R] = mu_train)
        mean_val = float(y_train.mean())
        pred_mean = np.full(n, mean_val)
        res_mean = self.evaluate_forecast(actual, pred_mean, "Historical Mean Baseline")

        # 3. Naive Persistence (y_hat_t = y_{t-1})
        pred_naive = np.roll(actual, 1)
        pred_naive[0] = y_train.iloc[-1] if len(y_train) > 0 else 0.0
        res_naive = self.evaluate_forecast(actual, pred_naive, "Naive Persistence Baseline")

        return {
            "Zero_Return": res_zero,
            "Historical_Mean": res_mean,
            "Naive_Persistence": res_naive,
        }
