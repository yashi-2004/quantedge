"""
QuantEdge Market Regime Detection Engine
Implements:
1. Gaussian Hidden Markov Models (HMM) via hmmlearn
2. Gaussian Mixture Models (GMM)
3. K-Means Volatility/Momentum Clustering
4. Rule-Based Regime Classification (Low-Vol Growth, High-Vol Decline, etc.)
Calculates transition matrices, expected regime durations, and conditional return distributions.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from hmmlearn.hmm import GaussianHMM
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


@dataclass
class RegimeDetectionResult:
    method: str
    n_regimes: int
    regime_labels: pd.Series  # 0, 1, ..., K-1 aligned to dates
    regime_names: Dict[int, str]  # Descriptive names e.g., 'Low-Vol Growth'
    transition_matrix: pd.DataFrame
    expected_durations_days: Dict[str, float]
    conditional_metrics: pd.DataFrame  # Return, Volatility, Sharpe per regime
    summary_text: str


class MarketRegimeDetector:
    """
    Identifies latent market states and structural shifts in financial time series.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def detect_regimes_hmm(
        self,
        returns: pd.Series,
        volatility: Optional[pd.Series] = None,
        n_regimes: int = 3,
    ) -> RegimeDetectionResult:
        """
        Fits a Gaussian Hidden Markov Model with full transition probability estimation.
        """
        r = returns.dropna()
        if volatility is None:
            vol = r.rolling(21, min_periods=21).std() * np.sqrt(252)
        else:
            vol = volatility.loc[r.index]

        df = pd.concat([r.rename("Return"), vol.rename("Volatility")], axis=1).dropna()
        if len(df) < 100:
            raise ValueError("Insufficient data for stable HMM regime fitting (need >= 100 rows).")

        scaler = StandardScaler()
        X = scaler.fit_transform(df.values)

        # Fit Gaussian HMM
        hmm = GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            n_iter=100,
            random_state=self.random_state,
        )
        hmm.fit(X)
        hidden_states = hmm.predict(X)

        # Sort regimes deterministically by volatility (0: lowest vol, ..., K-1: highest vol)
        state_vols = [df["Volatility"].iloc[hidden_states == k].mean() for k in range(n_regimes)]
        rank_map = {old: new for new, old in enumerate(np.argsort(state_vols))}
        sorted_states = np.array([rank_map[s] for s in hidden_states])

        # Transition matrix
        trans_mat_raw = hmm.transmat_
        # Permute transition matrix according to sorted ranks
        inv_map = {v: k for k, v in rank_map.items()}
        trans_mat_sorted = np.zeros((n_regimes, n_regimes))
        for i in range(n_regimes):
            for j in range(n_regimes):
                trans_mat_sorted[i, j] = trans_mat_raw[inv_map[i], inv_map[j]]

        # Human-readable labels
        regime_names_map = {
            0: "Low-Volatility Expansion (Bull)",
            1: "Moderate Volatility (Transitional)",
            2: "High-Volatility Contraction (Crisis / Bear)",
            3: "Extreme Shock Regime",
        }
        names = {k: regime_names_map.get(k, f"Regime {k}") for k in range(n_regimes)}

        # Expected duration: 1 / (1 - P_ii)
        expected_durations = {}
        for k in range(n_regimes):
            p_ii = trans_mat_sorted[k, k]
            dur = float(1.0 / (1.0 - p_ii + 1e-6))
            expected_durations[names[k]] = round(dur, 1)

        # Conditional Metrics
        metrics_list = []
        for k in range(n_regimes):
            mask = sorted_states == k
            r_k = df["Return"].iloc[mask]
            ann_ret = float(r_k.mean() * 252 * 100)
            ann_vol = float(r_k.std() * np.sqrt(252) * 100)
            sharpe = (ann_ret / (ann_vol + 1e-6)) if ann_vol > 0 else 0.0
            obs_pct = float(np.mean(mask) * 100)
            metrics_list.append(
                {
                    "Regime": names[k],
                    "Observations (%)": round(obs_pct, 1),
                    "Annual Return (%)": round(ann_ret, 2),
                    "Annual Volatility (%)": round(ann_vol, 2),
                    "Sharpe Ratio": round(sharpe, 2),
                }
            )

        cond_df = pd.DataFrame(metrics_list).set_index("Regime")
        trans_df = pd.DataFrame(
            trans_mat_sorted.round(4),
            index=[names[k] for k in range(n_regimes)],
            columns=[names[k] for k in range(n_regimes)],
        )

        regime_series = pd.Series(sorted_states, index=df.index, name="HMM_Regime")

        summary = (
            f"Fitted {n_regimes}-State Gaussian HMM. "
            f"State 0 captures lowest volatility ({cond_df.iloc[0]['Annual Volatility (%)']}% annualized), "
            f"while State {n_regimes-1} captures market turbulence ({cond_df.iloc[-1]['Annual Volatility (%)']}% annualized)."
        )

        return RegimeDetectionResult(
            method="Gaussian HMM",
            n_regimes=n_regimes,
            regime_labels=regime_series,
            regime_names=names,
            transition_matrix=trans_df,
            expected_durations_days=expected_durations,
            conditional_metrics=cond_df,
            summary_text=summary,
        )

    def detect_regimes_kmeans(self, returns: pd.Series, volatility: pd.Series, n_clusters: int = 3) -> RegimeDetectionResult:
        """
        K-Means clustering on return-volatility phase space.
        """
        df = pd.concat([returns.rename("Return"), volatility.rename("Volatility")], axis=1).dropna()
        scaler = StandardScaler()
        X = scaler.fit_transform(df.values)

        km = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        clusters = km.fit_predict(X)

        # Order clusters by volatility
        cluster_vols = [df["Volatility"].iloc[clusters == k].mean() for k in range(n_clusters)]
        rank_map = {old: new for new, old in enumerate(np.argsort(cluster_vols))}
        sorted_clusters = np.array([rank_map[c] for c in clusters])

        names = {
            0: "Low-Vol Regime (K-Means)",
            1: "Moderate-Vol Regime (K-Means)",
            2: "High-Vol Regime (K-Means)",
        }

        # Transition probabilities empirical
        trans_mat = np.zeros((n_clusters, n_clusters))
        for t in range(len(sorted_clusters) - 1):
            trans_mat[sorted_clusters[t], sorted_clusters[t + 1]] += 1
        row_sums = trans_mat.sum(axis=1, keepdims=True) + 1e-12
        trans_mat = trans_mat / row_sums

        expected_durations = {
            names[k]: round(float(1.0 / (1.0 - trans_mat[k, k] + 1e-6)), 1)
            for k in range(n_clusters)
        }

        metrics_list = []
        for k in range(n_clusters):
            mask = sorted_clusters == k
            r_k = df["Return"].iloc[mask]
            ann_ret = float(r_k.mean() * 252 * 100)
            ann_vol = float(r_k.std() * np.sqrt(252) * 100)
            sharpe = (ann_ret / (ann_vol + 1e-6)) if ann_vol > 0 else 0.0
            metrics_list.append(
                {
                    "Regime": names[k],
                    "Observations (%)": round(float(np.mean(mask) * 100), 1),
                    "Annual Return (%)": round(ann_ret, 2),
                    "Annual Volatility (%)": round(ann_vol, 2),
                    "Sharpe Ratio": round(sharpe, 2),
                }
            )

        cond_df = pd.DataFrame(metrics_list).set_index("Regime")
        trans_df = pd.DataFrame(
            trans_mat.round(4),
            index=[names[k] for k in range(n_clusters)],
            columns=[names[k] for k in range(n_clusters)],
        )

        regime_series = pd.Series(sorted_clusters, index=df.index, name="KMeans_Regime")

        return RegimeDetectionResult(
            method="K-Means Clustering",
            n_regimes=n_clusters,
            regime_labels=regime_series,
            regime_names=names,
            transition_matrix=trans_df,
            expected_durations_days=expected_durations,
            conditional_metrics=cond_df,
            summary_text="K-Means clustering on normalized returns and volatility.",
        )
