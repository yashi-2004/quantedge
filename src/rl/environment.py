"""
QuantEdge Gymnasium-Compatible Portfolio Environment
Simulates sequential asset allocation dynamics with transaction costs,
drawdown penalties, and zero lookahead leakage.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class PortfolioEnv:
    """
    Gymnasium-compatible sequential environment for quantitative portfolio allocation.
    """

    def __init__(
        self,
        returns_df: pd.DataFrame,
        window_size: int = 20,
        transaction_cost_bps: float = 10.0,
        risk_penalty: float = 2.0,
        drawdown_penalty: float = 1.0,
    ):
        self.returns_df = returns_df.copy().dropna()
        self.asset_names = list(returns_df.columns)
        self.n_assets = len(self.asset_names)
        self.window_size = window_size
        self.cost_rate = transaction_cost_bps / 10_000.0
        self.risk_penalty = risk_penalty
        self.drawdown_penalty = drawdown_penalty

        self.R = self.returns_df.values
        self.n_steps = len(self.R)

        # State space dimensions: (window_size, n_assets) flattened + current weights (n_assets,)
        self.obs_dim = (self.window_size * self.n_assets) + self.n_assets
        self.action_dim = self.n_assets

        self.reset()

    def reset(self) -> np.ndarray:
        self.current_step = self.window_size
        self.current_weights = np.ones(self.n_assets) / self.n_assets
        self.portfolio_value = 1.0
        self.peak_value = 1.0
        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        window_slice = self.R[self.current_step - self.window_size : self.current_step]
        flat_window = window_slice.flatten()
        obs = np.concatenate([flat_window, self.current_weights])
        return obs.astype(np.float32)

    def step(self, action_weights: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        # Normalize action weights to strictly adhere to simplex (sum = 1, w >= 0)
        action_weights = np.maximum(0.0, action_weights)
        total_w = np.sum(action_weights)
        if total_w > 0:
            target_weights = action_weights / total_w
        else:
            target_weights = np.ones(self.n_assets) / self.n_assets

        # Realize next period market return at current_step
        next_returns = self.R[self.current_step]
        gross_return = float(np.dot(target_weights, next_returns))

        # Turnover and transaction cost
        turnover = float(np.sum(np.abs(target_weights - self.current_weights)))
        cost = turnover * self.cost_rate
        net_return = gross_return - cost

        # Update portfolio value & peak
        self.portfolio_value *= 1.0 + net_return
        self.peak_value = max(self.peak_value, self.portfolio_value)
        current_drawdown = (self.peak_value - self.portfolio_value) / (self.peak_value + 1e-12)

        # Composite Reward: Sharpe-like differential with drawdown penalty
        # R_t = r_net - lambda_risk * r_net^2 - lambda_dd * Drawdown
        reward = net_return - (self.risk_penalty * (net_return**2)) - (self.drawdown_penalty * current_drawdown)

        # Drift weights
        denom = 1.0 + gross_return
        if denom > 0:
            self.current_weights = (target_weights * (1.0 + next_returns)) / denom
        else:
            self.current_weights = target_weights

        self.current_step += 1
        done = self.current_step >= (self.n_steps - 1)
        next_obs = self._get_observation() if not done else np.zeros(self.obs_dim, dtype=np.float32)

        info = {
            "net_return": net_return,
            "portfolio_value": self.portfolio_value,
            "turnover": turnover,
            "drawdown": current_drawdown,
            "weights": target_weights,
        }

        return next_obs, float(reward), done, info
