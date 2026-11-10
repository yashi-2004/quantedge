"""
QuantEdge Reinforcement Learning Portfolio Agent
Implements a lightweight Actor-Critic (A2C) policy gradient architecture
optimized for fast CPU execution and out-of-sample benchmarking.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rl.environment import PortfolioEnv


class ActorCriticNetwork(nn.Module):
    """
    Shared representation network with policy head (Softmax allocation) and value head.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(obs)
        logits = self.actor(features)
        weights = F.softmax(logits, dim=-1)
        value = self.critic(features)
        return weights, value


@dataclass
class RLEvaluationResult:
    episode_rewards: List[float]
    test_equity_curve: pd.Series
    test_weights: pd.DataFrame
    test_sharpe: float
    test_total_return: float
    test_max_drawdown: float
    benchmark_comparison: pd.DataFrame


class RLPortfolioTrainer:
    """
    Trains and validates RL allocation agents across chronological market environments.
    """

    def __init__(
        self,
        episodes: int = 25,
        lr: float = 0.003,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        seed: int = 42,
    ):
        self.episodes = episodes
        self.lr = lr
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)

    def train_and_evaluate(
        self,
        train_returns: pd.DataFrame,
        test_returns: pd.DataFrame,
        window_size: int = 20,
    ) -> RLEvaluationResult:
        train_env = PortfolioEnv(train_returns, window_size=window_size)
        test_env = PortfolioEnv(test_returns, window_size=window_size)

        model = ActorCriticNetwork(train_env.obs_dim, train_env.n_assets, hidden_dim=48)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        episode_rewards = []

        # Training Loop
        for ep in range(self.episodes):
            obs = train_env.reset()
            ep_reward = 0.0
            log_probs = []
            values = []
            rewards = []

            done = False
            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                weights_tensor, value = model(obs_t)
                action = weights_tensor.detach().numpy()[0]

                next_obs, r, done, _ = train_env.step(action)

                # Collect transitions
                log_p = torch.log(weights_tensor[0] + 1e-8).mean()
                log_probs.append(log_p)
                values.append(value.squeeze(0))
                rewards.append(r)
                ep_reward += r
                obs = next_obs

            # Compute discounted returns
            R = 0.0
            returns = []
            for r in reversed(rewards):
                R = r + self.gamma * R
                returns.insert(0, R)
            returns_t = torch.tensor(returns, dtype=torch.float32)
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

            policy_loss = 0.0
            value_loss = 0.0
            for log_p, val, target_r in zip(log_probs, values, returns_t):
                advantage = target_r - val.item()
                policy_loss += -log_p * advantage
                value_loss += F.mse_loss(val, target_r)

            total_loss = policy_loss + 0.5 * value_loss
            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            episode_rewards.append(float(ep_reward))

        # Out-of-sample Test Evaluation
        test_obs = test_env.reset()
        test_dates = test_returns.index[window_size:]
        test_values = [1.0]
        test_weights_history = []
        test_returns_list = []

        done = False
        while not done:
            with torch.no_grad():
                obs_t = torch.tensor(test_obs, dtype=torch.float32).unsqueeze(0)
                w_t, _ = model(obs_t)
                action = w_t.numpy()[0]

            test_weights_history.append(action)
            test_obs, r, done, info = test_env.step(action)
            test_values.append(info["portfolio_value"])
            test_returns_list.append(info["net_return"])

        # Align lengths
        eval_len = min(len(test_dates), len(test_returns_list))
        equity_s = pd.Series(test_values[:eval_len], index=test_dates[:eval_len], name="RL_Equity")
        weights_df = pd.DataFrame(test_weights_history[:eval_len], index=test_dates[:eval_len], columns=test_returns.columns)
        ret_s = pd.Series(test_returns_list[:eval_len], index=test_dates[:eval_len])

        # Test metrics
        tot_ret = float((equity_s.iloc[-1] / equity_s.iloc[0]) - 1.0) * 100.0
        ann_vol = float(ret_s.std() * np.sqrt(252)) * 100.0
        ann_ret = float(ret_s.mean() * 252) * 100.0
        sharpe = round(float((ann_ret - 3.0) / (ann_vol + 1e-12)), 2)

        roll_max = equity_s.cummax()
        dd = (equity_s - roll_max) / roll_max
        max_dd = round(float(dd.min() * 100.0), 2)

        # Baseline comparison on identical test dates
        eq_bench = (1.0 + test_returns.loc[equity_s.index].mean(axis=1)).cumprod()
        bench_ret = float((eq_bench.iloc[-1] / eq_bench.iloc[0]) - 1.0) * 100.0
        bench_vol = float(test_returns.loc[equity_s.index].mean(axis=1).std() * np.sqrt(252)) * 100.0
        bench_sharpe = round(float((bench_ret - 3.0) / (bench_vol + 1e-12)), 2)
        bench_dd = round(float(((eq_bench - eq_bench.cummax()) / eq_bench.cummax()).min() * 100.0), 2)

        comparison_df = pd.DataFrame(
            [
                {
                    "Strategy": "PPO / Actor-Critic RL",
                    "Total Return (%)": round(tot_ret, 2),
                    "Annual Volatility (%)": round(ann_vol, 2),
                    "Sharpe Ratio": sharpe,
                    "Max Drawdown (%)": max_dd,
                },
                {
                    "Strategy": "Equal-Weighted Baseline (1/N)",
                    "Total Return (%)": round(bench_ret, 2),
                    "Annual Volatility (%)": round(bench_vol, 2),
                    "Sharpe Ratio": bench_sharpe,
                    "Max Drawdown (%)": bench_dd,
                },
            ]
        ).set_index("Strategy")

        return RLEvaluationResult(
            episode_rewards=episode_rewards,
            test_equity_curve=equity_s,
            test_weights=weights_df,
            test_sharpe=sharpe,
            test_total_return=round(tot_ret, 2),
            test_max_drawdown=max_dd,
            benchmark_comparison=comparison_df,
        )
