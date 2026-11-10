"""
QuantEdge Kalman Filter Engine
Implements recursive state-space Kalman filters for:
1. Microstructure noise filtering / latent fundamental trend extraction.
2. Dynamic, time-varying hedge ratio & beta tracking for pairs trading and asset allocation.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class KalmanTrendResult:
    filtered_trend: pd.Series
    estimated_noise_std: float
    state_variance: pd.Series
    residuals: pd.Series


@dataclass
class KalmanBetaResult:
    time_varying_alpha: pd.Series
    time_varying_beta: pd.Series
    spread_residuals: pd.Series
    state_covariance: pd.DataFrame


class KalmanFilterEngine:
    """
    Optimal recursive Bayesian estimator for dynamic linear Gaussian state-space systems.
    """

    def filter_price_trend(
        self,
        prices: pd.Series,
        observation_noise_var: float = 1.0,
        process_noise_var: float = 0.01,
    ) -> KalmanTrendResult:
        """
        Latent trend model:
        Observation: y_t = x_t + v_t,  v_t ~ N(0, R)
        State transition: x_t = x_{t-1} + w_t,  w_t ~ N(0, Q)
        """
        y = prices.dropna().values
        n = len(y)
        if n < 5:
            raise ValueError("Series too short for Kalman filtering.")

        R = observation_noise_var
        Q = process_noise_var

        # Initial state prior
        x_hat = np.zeros(n)
        P = np.zeros(n)
        x_hat[0] = y[0]
        P[0] = 1.0

        for t in range(1, n):
            # 1. Predict
            x_pred = x_hat[t - 1]
            P_pred = P[t - 1] + Q

            # 2. Update
            innovation = y[t] - x_pred
            S = P_pred + R  # Innovation covariance
            K = P_pred / S  # Kalman gain

            x_hat[t] = x_pred + K * innovation
            P[t] = (1.0 - K) * P_pred

        idx = prices.dropna().index
        trend_s = pd.Series(x_hat, index=idx, name="Kalman_Trend")
        var_s = pd.Series(P, index=idx, name="Estimation_Variance")
        resids = pd.Series(y - x_hat, index=idx, name="Noise_Residuals")

        return KalmanTrendResult(
            filtered_trend=trend_s,
            estimated_noise_std=float(np.sqrt(np.mean((y - x_hat) ** 2))),
            state_variance=var_s,
            residuals=resids,
        )

    def dynamic_hedge_ratio(
        self,
        asset_y: pd.Series,
        asset_x: pd.Series,
        delta_noise: float = 1e-4,
        R_obs: float = 1e-3,
    ) -> KalmanBetaResult:
        """
        Estimates time-varying alpha and beta:
        y_t = [1, x_t] * [alpha_t, beta_t]^T + e_t
        State transitions as random walk with covariance Q = (delta / (1 - delta)) * I
        """
        df = pd.concat([asset_y, asset_x], axis=1).dropna()
        if len(df) < 10:
            raise ValueError("Insufficient paired observations for dynamic hedge ratio estimation.")

        y = df.iloc[:, 0].values
        x = df.iloc[:, 1].values
        n = len(y)

        # State vector: [alpha, beta]
        theta = np.zeros((n, 2))
        theta[0] = [0.0, 1.0]

        P = np.eye(2) * 1.0
        Vw = (delta_noise / (1.0 - delta_noise)) * np.eye(2)
        Ve = R_obs

        alphas = np.zeros(n)
        betas = np.zeros(n)
        spreads = np.zeros(n)

        for t in range(n):
            if t > 0:
                # State prediction
                theta[t] = theta[t - 1]
                P = P + Vw

            # Observation matrix H = [1, x_t]
            H = np.array([1.0, x[t]])
            y_pred = np.dot(H, theta[t])
            err = y[t] - y_pred

            # Kalman gain
            S = float(np.dot(H, np.dot(P, H.T))) + Ve
            K = np.dot(P, H.T) / S

            # State update
            theta[t] = theta[t] + K * err
            P = P - np.outer(K, H).dot(P)

            alphas[t] = theta[t, 0]
            betas[t] = theta[t, 1]
            spreads[t] = err

        idx = df.index
        return KalmanBetaResult(
            time_varying_alpha=pd.Series(alphas, index=idx, name="Kalman_Alpha"),
            time_varying_beta=pd.Series(betas, index=idx, name="Kalman_Beta"),
            spread_residuals=pd.Series(spreads, index=idx, name="Kalman_Spread"),
            state_covariance=pd.DataFrame(P, index=["Alpha", "Beta"], columns=["Alpha", "Beta"]),
        )
