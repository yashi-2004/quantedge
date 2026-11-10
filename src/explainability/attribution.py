"""
QuantEdge Model Explainability & Attribution Engine
Implements:
1. Global and local feature attributions
2. Feature ablation experiments (measuring marginal contribution of feature groups)
3. Multicollinearity caveat analysis (distinguishing correlation from causal drivers)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error


@dataclass
class AblationResult:
    feature_group: str
    num_features: int
    test_rmse: float
    test_directional_accuracy: float
    delta_rmse_vs_all: float  # Positive delta means performance degraded when removed (i.e., feature group was helpful)


@dataclass
class LocalExplanation:
    date: str
    predicted_return: float
    actual_return: float
    top_positive_contributors: Dict[str, float]
    top_negative_contributors: Dict[str, float]


class ModelExplainer:
    """
    Computes global importance, local attributions, and group ablation benchmarks.
    """

    def compute_permutation_importance(
        self, model: Any, X_test: pd.DataFrame, y_test: pd.Series, n_repeats: int = 5
    ) -> pd.Series:
        perm = permutation_importance(model, X_test, y_test, n_repeats=n_repeats, random_state=42)
        importances = pd.Series(perm.importances_mean, index=X_test.columns).sort_values(ascending=False)
        return importances.round(6)

    def explain_instance(
        self, model: Any, X_row: pd.Series, feature_names: List[str], baseline_row: pd.Series
    ) -> Dict[str, float]:
        """
        Computes linear or tree feature attribution approximation for a single prediction point.
        """
        if hasattr(model, "coef_"):
            coefs = model.coef_
            diff = (X_row.values - baseline_row.values) * coefs
            attributions = pd.Series(diff, index=feature_names).sort_values(ascending=False)
        elif hasattr(model, "feature_importances_"):
            # Tree surrogate attribution: direction scaled by relative importance
            diff = X_row.values - baseline_row.values
            weights = model.feature_importances_
            signed_impact = diff * weights
            attributions = pd.Series(signed_impact, index=feature_names).sort_values(ascending=False)
        else:
            attributions = pd.Series(0.0, index=feature_names)

        return attributions.to_dict()

    def run_feature_ablation_study(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        feature_groups: Dict[str, List[str]],
    ) -> List[AblationResult]:
        """
        Trains model leaving out one feature group at a time to isolate empirical utility.
        """
        from sklearn.linear_model import Ridge

        # Baseline: All features
        base_mod = Ridge(alpha=1.0)
        base_mod.fit(X_train, y_train)
        pred_all = base_mod.predict(X_test)
        base_rmse = float(np.sqrt(mean_squared_error(y_test, pred_all)))
        base_dir_acc = float(np.mean((y_test.values > 0) == (pred_all > 0)) * 100.0)

        results = [
            AblationResult(
                feature_group="ALL_FEATURES (Baseline)",
                num_features=X_train.shape[1],
                test_rmse=round(base_rmse, 6),
                test_directional_accuracy=round(base_dir_acc, 2),
                delta_rmse_vs_all=0.0,
            )
        ]

        for group_name, cols in feature_groups.items():
            cols_to_drop = [c for c in cols if c in X_train.columns]
            if not cols_to_drop:
                continue

            X_tr_sub = X_train.drop(columns=cols_to_drop)
            X_te_sub = X_test.drop(columns=cols_to_drop)

            mod = Ridge(alpha=1.0)
            mod.fit(X_tr_sub, y_train)
            preds = mod.predict(X_te_sub)
            sub_rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
            sub_dir_acc = float(np.mean((y_test.values > 0) == (preds > 0)) * 100.0)
            delta_rmse = sub_rmse - base_rmse

            results.append(
                AblationResult(
                    feature_group=f"WITHOUT: {group_name}",
                    num_features=X_tr_sub.shape[1],
                    test_rmse=round(sub_rmse, 6),
                    test_directional_accuracy=round(sub_dir_acc, 2),
                    delta_rmse_vs_all=round(delta_rmse, 6),
                )
            )

        return results
