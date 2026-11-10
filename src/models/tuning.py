"""
QuantEdge Hyperparameter Optimization Engine
Uses Optuna with strict TimeSeriesSplit cross-validation to tune classical and ensemble models.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except (ImportError, Exception):
    HAS_XGBOOST = False

optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class TuningResult:
    best_params: Dict[str, Any]
    best_score: float
    metric_name: str
    n_trials: int
    history_df: pd.DataFrame


class HyperparameterTuner:
    """
    Automated Bayesian parameter optimization with chronological cross-validation.
    """

    def __init__(self, n_splits: int = 3, n_trials: int = 15, random_state: int = 42):
        self.n_splits = n_splits
        self.n_trials = n_trials
        self.random_state = random_state

    def tune_xgboost(self, X: pd.DataFrame, y: pd.Series) -> TuningResult:
        tscv = TimeSeriesSplit(n_splits=self.n_splits)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 30, 150, step=20),
                "max_depth": trial.suggest_int("max_depth", 2, 6),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                "random_state": self.random_state,
                "verbosity": 0,
                "n_jobs": 1,
            }

            cv_scores = []
            for train_idx, val_idx in tscv.split(X):
                X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
                X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]

                if HAS_XGBOOST:
                    model = xgb.XGBRegressor(**params)
                else:
                    model = HistGradientBoostingRegressor(
                        max_iter=params["n_estimators"],
                        max_depth=params["max_depth"],
                        learning_rate=params["learning_rate"],
                        random_state=self.random_state,
                    )
                model.fit(X_tr, y_tr)
                preds = model.predict(X_va)
                rmse = np.sqrt(np.mean((y_va.values - preds) ** 2))
                cv_scores.append(rmse)

            return float(np.mean(cv_scores))

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials)

        history = [
            {"trial": t.number, "value": t.value, **t.params}
            for t in study.trials
            if t.value is not None
        ]

        return TuningResult(
            best_params=study.best_params,
            best_score=round(float(study.best_value), 6),
            metric_name="CV_RMSE",
            n_trials=len(study.trials),
            history_df=pd.DataFrame(history),
        )
