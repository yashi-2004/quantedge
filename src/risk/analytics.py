"""
QuantEdge Advanced Risk Management & Stress Testing Laboratory
Implements research-grade risk analytics:
1. Value at Risk (Historical, Parametric Gaussian/Student-t, Monte Carlo)
2. Conditional Value at Risk (CVaR / Expected Shortfall)
3. Monte Carlo Multi-Asset Correlated Path Simulation
4. Historical and Hypothetical Macro Stress Testing
5. Risk Decomposition and Portfolio Concentration Metrics
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class VaRReport:
    confidence_level: float
    historical_var_pct: float
    parametric_var_pct: float
    student_t_var_pct: float
    monte_carlo_var_pct: float
    historical_cvar_pct: float
    parametric_cvar_pct: float
    dollar_var: float
    dollar_cvar: float


@dataclass
class MonteCarloSimulationResult:
    horizon_days: int
    num_simulations: int
    terminal_values: np.ndarray
    simulated_paths: np.ndarray  # Shape (num_paths, horizon)
    percentiles: Dict[str, float]
    probability_of_loss: float
    expected_shortfall_95: float


@dataclass
class StressTestResult:
    scenario_name: str
    portfolio_impact_pct: float
    dollar_loss: float
    asset_shocks: Dict[str, float]
    is_hypothetical: bool


class RiskLaboratory:
    """
    Computes tail risk, simulates future paths, and tests severe macroeconomic stress scenarios.
    """

    HISTORICAL_SCENARIOS = {
        "2008 Global Financial Crisis": {
            "SPY": -0.42, "QQQ": -0.40, "AAPL": -0.45, "NVDA": -0.55,
            "JPM": -0.35, "GLD": 0.15, "TLT": 0.22, "RELIANCE.NS": -0.48,
        },
        "2020 COVID Market Shock": {
            "SPY": -0.34, "QQQ": -0.28, "AAPL": -0.25, "NVDA": -0.20,
            "JPM": -0.38, "GLD": -0.05, "TLT": 0.18, "RELIANCE.NS": -0.32,
        },
        "2022 Inflation & Rate Shock": {
            "SPY": -0.19, "QQQ": -0.33, "AAPL": -0.27, "NVDA": -0.50,
            "JPM": -0.15, "GLD": -0.01, "TLT": -0.31, "RELIANCE.NS": 0.05,
        },
    }

    def __init__(self, portfolio_value: float = 100_000.0):
        self.portfolio_value = portfolio_value

    def calculate_var_cvar(
        self,
        portfolio_returns: pd.Series,
        confidence_level: float = 0.95,
        horizon_days: int = 1,
    ) -> VaRReport:
        r = portfolio_returns.dropna().values
        if len(r) < 30:
            raise ValueError("Insufficient returns for statistical VaR estimation.")

        alpha = 1.0 - confidence_level

        # 1. Historical VaR & CVaR
        hist_var = float(-np.percentile(r, alpha * 100.0) * np.sqrt(horizon_days))
        tail_losses = r[r <= -hist_var / np.sqrt(horizon_days)]
        hist_cvar = float(-tail_losses.mean() * np.sqrt(horizon_days)) if len(tail_losses) > 0 else hist_var

        # 2. Parametric Gaussian VaR & CVaR
        mu = float(np.mean(r))
        sigma = float(np.std(r))
        z_score = stats.norm.ppf(confidence_level)
        param_var = float(-(mu - z_score * sigma) * np.sqrt(horizon_days))
        # Parametric CVaR: mu + sigma * phi(z) / alpha
        phi_z = stats.norm.pdf(z_score)
        param_cvar = float(-(mu - sigma * (phi_z / alpha)) * np.sqrt(horizon_days))

        # 3. Student-t VaR (Fat Tails)
        df, loc, scale = stats.t.fit(r)
        t_crit = stats.t.ppf(alpha, df=df)
        student_t_var = float(-(loc + scale * t_crit) * np.sqrt(horizon_days))

        # 4. Monte Carlo VaR (10,000 empirical bootstrap draws)
        np.random.seed(42)
        mc_draws = np.random.choice(r, size=(10000, horizon_days)).sum(axis=1)
        mc_var = float(-np.percentile(mc_draws, alpha * 100.0))

        dollar_var = round(self.portfolio_value * (hist_var / 100.0), 2)
        dollar_cvar = round(self.portfolio_value * (hist_cvar / 100.0), 2)

        return VaRReport(
            confidence_level=confidence_level,
            historical_var_pct=round(hist_var * 100.0, 2),
            parametric_var_pct=round(param_var * 100.0, 2),
            student_t_var_pct=round(student_t_var * 100.0, 2),
            monte_carlo_var_pct=round(mc_var * 100.0, 2),
            historical_cvar_pct=round(hist_cvar * 100.0, 2),
            parametric_cvar_pct=round(param_cvar * 100.0, 2),
            dollar_var=dollar_var,
            dollar_cvar=dollar_cvar,
        )

    def monte_carlo_portfolio_simulation(
        self,
        returns_df: pd.DataFrame,
        weights: np.ndarray,
        horizon_days: int = 63,
        num_simulations: int = 2000,
        seed: int = 42,
    ) -> MonteCarloSimulationResult:
        """
        Simulates future portfolio values via correlated Geometric Brownian Motion:
        dS_i / S_i = mu_i * dt + L * dW_t
        where L is the Cholesky factor of the return covariance matrix.
        """
        np.random.seed(seed)
        mu = returns_df.mean().values
        cov = returns_df.cov().values

        # Ensure positive semi-definite covariance for Cholesky
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        eigenvals = np.maximum(eigenvals, 1e-8)
        cov_clean = eigenvecs.dot(np.diag(eigenvals)).dot(eigenvecs.T)
        L = np.linalg.cholesky(cov_clean)

        n_assets = len(weights)
        paths = np.zeros((num_simulations, horizon_days + 1))
        paths[:, 0] = self.portfolio_value

        for sim in range(num_simulations):
            # Standard normal random shocks
            z = np.random.normal(size=(horizon_days, n_assets))
            correlated_innovations = z.dot(L.T)  # (horizon_days, n_assets)
            daily_asset_returns = mu + correlated_innovations
            daily_port_returns = daily_asset_returns.dot(weights)

            # Cumulative portfolio trajectory
            paths[sim, 1:] = self.portfolio_value * np.cumprod(1.0 + daily_port_returns)

        term_vals = paths[:, -1]
        pcts = {
            "5th Percentile (Severe Bear)": round(float(np.percentile(term_vals, 5)), 2),
            "25th Percentile (Moderate Bear)": round(float(np.percentile(term_vals, 25)), 2),
            "50th Percentile (Median Outlook)": round(float(np.percentile(term_vals, 50)), 2),
            "75th Percentile (Moderate Bull)": round(float(np.percentile(term_vals, 75)), 2),
            "95th Percentile (Strong Bull)": round(float(np.percentile(term_vals, 95)), 2),
        }

        prob_loss = float(np.mean(term_vals < self.portfolio_value) * 100.0)
        cvar_95_val = float(np.percentile(term_vals, 5))
        tail_terminals = term_vals[term_vals <= cvar_95_val]
        expected_shortfall = round(float(self.portfolio_value - tail_terminals.mean()), 2)

        return MonteCarloSimulationResult(
            horizon_days=horizon_days,
            num_simulations=num_simulations,
            terminal_values=term_vals,
            simulated_paths=paths[:150],  # Save 150 representative paths for plotting
            percentiles=pcts,
            probability_of_loss=round(prob_loss, 2),
            expected_shortfall_95=expected_shortfall,
        )

    def run_stress_test(
        self,
        weights_series: pd.Series,
        scenario_name: str,
        custom_market_shock: Optional[float] = None,
    ) -> StressTestResult:
        """
        Applies historical or hypothetical macro stress scenarios.
        """
        weights = weights_series.to_dict()
        shocks: Dict[str, float] = {}

        if scenario_name in self.HISTORICAL_SCENARIOS:
            hist_shocks = self.HISTORICAL_SCENARIOS[scenario_name]
            for ticker in weights.keys():
                shocks[ticker] = hist_shocks.get(ticker, -0.25)
            is_hypo = False
        else:
            # Hypothetical shock
            shock_pct = custom_market_shock if custom_market_shock is not None else -0.20
            for ticker in weights.keys():
                if "GLD" in ticker or "TLT" in ticker:
                    shocks[ticker] = 0.05  # Safe haven hedge
                else:
                    shocks[ticker] = shock_pct
            is_hypo = True

        port_impact = sum(weights[t] * shocks[t] for t in weights.keys())
        dollar_loss = self.portfolio_value * port_impact

        return StressTestResult(
            scenario_name=scenario_name,
            portfolio_impact_pct=round(port_impact * 100.0, 2),
            dollar_loss=round(dollar_loss, 2),
            asset_shocks={k: round(v * 100.0, 1) for k, v in shocks.items()},
            is_hypothetical=is_hypo,
        )
