"""
QuantEdge Classical Machine Learning Suite
Implements regularized linear models, tree ensembles, and XGBoost
strictly under chronological time-series evaluation.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except (ImportError, Exception):
    HAS_XGBOOST = False

from src.models.baseline import FinancialBaselines


@dataclass
class ModelEvaluationResult:
    model_name: str
    task_type: str  # 'regression' or 'classification'
    in_sample_metrics: Dict[str, float]
    out_of_sample_metrics: Dict[str, float]
    predictions_test: pd.Series
    actuals_test: pd.Series
    feature_importances: pd.Series
    fitted_model: Any
    scaler: Optional[StandardScaler]


class ClassicalMLSuite:
    """
    Unified training and evaluation pipeline for classical quantitative ML algorithms.
    """

    SUPPORTED_REGRESSION = [
        "LinearRegression",
        "Ridge",
        "Lasso",
        "ElasticNet",
        "RandomForest",
        "GradientBoosting",
        "XGBoost",
    ]

    SUPPORTED_CLASSIFICATION = [
        "LogisticRegression",
        "RandomForestClassifier",
        "GradientBoostingClassifier",
        "XGBoostClassifier",
    ]

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def chronological_split(
        self, X: pd.DataFrame, y: pd.Series, train_ratio: float = 0.70, val_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Enforces chronological temporal split. Never randomly shuffles time series!
        """
        n = len(X)
        n_train = int(n * train_ratio)
        n_val = int(n * (train_ratio + val_ratio))

        X_train, y_train = X.iloc[:n_train], y.iloc[:n_train]
        X_val, y_val = X.iloc[n_train:n_val], y.iloc[n_train:n_val]
        X_test, y_test = X.iloc[n_val:], y.iloc[n_val:]

        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_regression_model(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        params: Optional[Dict[str, Any]] = None,
        scale_features: bool = True,
    ) -> ModelEvaluationResult:
        params = params or {}
        scaler = StandardScaler() if scale_features else None

        if scaler:
            X_tr = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
            X_te = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
        else:
            X_tr, X_te = X_train, X_test

        # Instantiate model
        if model_name == "LinearRegression":
            model = LinearRegression()
        elif model_name == "Ridge":
            alpha = params.get("alpha", 1.0)
            model = Ridge(alpha=alpha, random_state=self.random_state)
        elif model_name == "Lasso":
            alpha = params.get("alpha", 0.001)
            model = Lasso(alpha=alpha, random_state=self.random_state, max_iter=2000)
        elif model_name == "ElasticNet":
            alpha = params.get("alpha", 0.001)
            l1_ratio = params.get("l1_ratio", 0.5)
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=self.random_state)
        elif model_name == "RandomForest":
            n_est = params.get("n_estimators", 100)
            max_d = params.get("max_depth", 5)
            model = RandomForestRegressor(n_estimators=n_est, max_depth=max_d, random_state=self.random_state, n_jobs=-1)
        elif model_name == "GradientBoosting":
            n_est = params.get("n_estimators", 100)
            lr = params.get("learning_rate", 0.05)
            model = GradientBoostingRegressor(n_estimators=n_est, learning_rate=lr, max_depth=3, random_state=self.random_state)
        elif model_name == "XGBoost":
            n_est = params.get("n_estimators", 100)
            lr = params.get("learning_rate", 0.05)
            max_d = params.get("max_depth", 3)
            if HAS_XGBOOST:
                model = xgb.XGBRegressor(
                    n_estimators=n_est,
                    learning_rate=lr,
                    max_depth=max_d,
                    random_state=self.random_state,
                    n_jobs=-1,
                    verbosity=0,
                )
            else:
                model = HistGradientBoostingRegressor(
                    max_iter=n_est,
                    learning_rate=lr,
                    max_depth=max_d,
                    random_state=self.random_state,
                )
        else:
            raise ValueError(f"Unsupported regression model: {model_name}")

        model.fit(X_tr, y_train)

        # Predictions
        pred_train = model.predict(X_tr)
        pred_test = model.predict(X_te)
        pred_series = pd.Series(pred_test, index=y_test.index, name=f"{model_name}_Pred")

        # Evaluate via FinancialBaselines
        m_train = FinancialBaselines.evaluate_forecast(y_train.values, pred_train, f"{model_name}_Train")
        m_test = FinancialBaselines.evaluate_forecast(y_test.values, pred_test, f"{model_name}_Test")

        in_sample = {
            "RMSE": m_train.rmse,
            "MAE": m_train.mae,
            "Directional_Accuracy": m_train.directional_accuracy,
            "IC": m_train.information_coefficient,
            "R2": m_train.r_squared,
        }
        out_sample = {
            "RMSE": m_test.rmse,
            "MAE": m_test.mae,
            "Directional_Accuracy": m_test.directional_accuracy,
            "IC": m_test.information_coefficient,
            "R2": m_test.r_squared,
        }

        # Feature importances
        if hasattr(model, "coef_"):
            importances = pd.Series(np.abs(model.coef_), index=X_train.columns).sort_values(ascending=False)
        elif hasattr(model, "feature_importances_"):
            importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
        else:
            importances = pd.Series(0.0, index=X_train.columns)

        return ModelEvaluationResult(
            model_name=model_name,
            task_type="regression",
            in_sample_metrics=in_sample,
            out_of_sample_metrics=out_sample,
            predictions_test=pred_series,
            actuals_test=y_test,
            feature_importances=importances,
            fitted_model=model,
            scaler=scaler,
        )

    def train_classification_model(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        params: Optional[Dict[str, Any]] = None,
        scale_features: bool = True,
    ) -> ModelEvaluationResult:
        params = params or {}
        scaler = StandardScaler() if scale_features else None

        if scaler:
            X_tr = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
            X_te = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
        else:
            X_tr, X_te = X_train, X_test

        if model_name == "LogisticRegression":
            c_val = params.get("C", 1.0)
            model = LogisticRegression(C=c_val, random_state=self.random_state, max_iter=1000)
        elif model_name == "RandomForestClassifier":
            n_est = params.get("n_estimators", 100)
            max_d = params.get("max_depth", 4)
            model = RandomForestClassifier(n_estimators=n_est, max_depth=max_d, random_state=self.random_state, n_jobs=-1)
        elif model_name == "GradientBoostingClassifier":
            n_est = params.get("n_estimators", 100)
            lr = params.get("learning_rate", 0.05)
            model = GradientBoostingClassifier(n_estimators=n_est, learning_rate=lr, max_depth=3, random_state=self.random_state)
        elif model_name == "XGBoostClassifier":
            n_est = params.get("n_estimators", 100)
            lr = params.get("learning_rate", 0.05)
            max_d = params.get("max_depth", 3)
            if HAS_XGBOOST:
                model = xgb.XGBClassifier(
                    n_estimators=n_est,
                    learning_rate=lr,
                    max_depth=max_d,
                    random_state=self.random_state,
                    n_jobs=-1,
                    verbosity=0,
                )
            else:
                model = HistGradientBoostingClassifier(
                    max_iter=n_est,
                    learning_rate=lr,
                    max_depth=max_d,
                    random_state=self.random_state,
                )
        else:
            raise ValueError(f"Unsupported classification model: {model_name}")

        model.fit(X_tr, y_train)

        pred_tr = model.predict(X_tr)
        pred_te = model.predict(X_te)
        prob_te = model.predict_proba(X_te)[:, 1] if hasattr(model, "predict_proba") else pred_te

        auc_tr = float(roc_auc_score(y_train, model.predict_proba(X_tr)[:, 1])) if hasattr(model, "predict_proba") else 0.5
        auc_te = float(roc_auc_score(y_test, prob_te)) if hasattr(model, "predict_proba") else 0.5

        in_sample = {
            "Accuracy": round(float(accuracy_score(y_train, pred_tr) * 100), 2),
            "Precision": round(float(precision_score(y_train, pred_tr, zero_division=0) * 100), 2),
            "Recall": round(float(recall_score(y_train, pred_tr, zero_division=0) * 100), 2),
            "F1": round(float(f1_score(y_train, pred_tr, zero_division=0)), 4),
            "ROC_AUC": round(auc_tr, 4),
        }
        out_sample = {
            "Accuracy": round(float(accuracy_score(y_test, pred_te) * 100), 2),
            "Precision": round(float(precision_score(y_test, pred_te, zero_division=0) * 100), 2),
            "Recall": round(float(recall_score(y_test, pred_te, zero_division=0) * 100), 2),
            "F1": round(float(f1_score(y_test, pred_te, zero_division=0)), 4),
            "ROC_AUC": round(auc_te, 4),
        }

        # Feature importances
        if hasattr(model, "coef_"):
            importances = pd.Series(np.abs(model.coef_[0]), index=X_train.columns).sort_values(ascending=False)
        elif hasattr(model, "feature_importances_"):
            importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
        else:
            importances = pd.Series(0.0, index=X_train.columns)

        pred_series = pd.Series(prob_te, index=y_test.index, name=f"{model_name}_Prob")

        return ModelEvaluationResult(
            model_name=model_name,
            task_type="classification",
            in_sample_metrics=in_sample,
            out_of_sample_metrics=out_sample,
            predictions_test=pred_series,
            actuals_test=y_test,
            feature_importances=importances,
            fitted_model=model,
            scaler=scaler,
        )
