"""
QuantEdge Portfolio Optimization Engine
Implements mathematically rigorous portfolio allocation models:
1. Equal-Weighted Baseline
2. Markowitz Mean-Variance (Max Sharpe, Min Variance, Efficient Frontier)
3. Risk Parity (Equal Risk Contribution via SLSQP)
4. Black-Litterman Model (Equilibrium Returns, Investor Views, Posterior Estimation)
5. CVaR (Expected Shortfall / Tail Risk) Optimization
6. Risk Decomposition (MCR, Percentage Contribution to Risk, Diversification Ratio)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class PortfolioResult:
    method: str
    weights: pd.Series
    expected_return_annual: float
    volatility_annual: float
    sharpe_ratio: float
    sortino_ratio: float
    cvar_95: float
    marginal_risk_contributions: pd.Series
    percentage_risk_contributions: pd.Series
    diversification_ratio: float


@dataclass
class EfficientFrontierResult:
    target_returns: np.ndarray
    target_volatilities: np.ndarray
    weights_matrix: np.ndarray
    sharpe_ratios: np.ndarray
    tickers: List[str]


class PortfolioOptimizer:
    """
    Quantitative portfolio allocation engine with configurable convex and non-linear constraints.
    """

    def __init__(self, risk_free_rate: float = 0.03):
        self.rf = risk_free_rate

    def _annualized_stats(
        self, returns_df: pd.DataFrame
    ) -> Tuple[pd.Series, pd.DataFrame]:
        mean_ret = returns_df.mean() * 252
        cov_mat = returns_df.cov() * 252
        return mean_ret, cov_mat

    def risk_decomposition(
        self, weights: np.ndarray, cov_mat: np.ndarray, tickers: List[str]
    ) -> Tuple[pd.Series, pd.Series, float]:
        """
        Calculates Marginal Contribution to Risk (MCR) and % Contribution to Risk (PCR).
        MCR_i = (Sigma * w)_i / sigma_p
        PCR_i = w_i * MCR_i / sigma_p
        """
        port_var = float(np.dot(weights.T, np.dot(cov_mat, weights)))
        port_vol = float(np.sqrt(max(1e-12, port_var)))
        mcr = np.dot(cov_mat, weights) / port_vol
        pcr = (weights * mcr) / port_vol

        # Diversification Ratio = (w^T * sigmas) / sigma_p
        asset_vols = np.sqrt(np.diag(cov_mat))
        div_ratio = float(np.dot(weights, asset_vols) / port_vol) if port_vol > 0 else 1.0

        return (
            pd.Series(mcr, index=tickers, name="MCR"),
            pd.Series(pcr * 100.0, index=tickers, name="PCR_Pct"),
            round(div_ratio, 2),
        )

    def calculate_cvar(self, returns_df: pd.DataFrame, weights: np.ndarray, alpha: float = 0.95) -> float:
        port_returns = returns_df.dot(weights)
        var_alpha = float(np.percentile(port_returns, (1.0 - alpha) * 100))
        tail_losses = port_returns[port_returns <= var_alpha]
        cvar = float(-tail_losses.mean() * np.sqrt(252)) if len(tail_losses) > 0 else 0.0
        return round(cvar, 4)

    def calculate_sortino(self, returns_df: pd.DataFrame, weights: np.ndarray) -> float:
        port_returns = returns_df.dot(weights)
        downside_returns = port_returns[port_returns < (self.rf / 252)]
        downside_std = float(downside_returns.std() * np.sqrt(252)) if len(downside_returns) > 1 else 1e-6
        ann_return = float(port_returns.mean() * 252)
        sortino = (ann_return - self.rf) / downside_std
        return round(sortino, 2)

    # 1. Equal-Weighted Baseline
    def equal_weighted(self, returns_df: pd.DataFrame) -> PortfolioResult:
        n = returns_df.shape[1]
        tickers = list(returns_df.columns)
        weights = np.ones(n) / n
        mean_ret, cov_mat = self._annualized_stats(returns_df)

        exp_ret = float(np.dot(weights, mean_ret))
        port_vol = float(np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights))))
        sharpe = (exp_ret - self.rf) / (port_vol + 1e-12)
        cvar = self.calculate_cvar(returns_df, weights)
        sortino = self.calculate_sortino(returns_df, weights)
        mcr, pcr, div_ratio = self.risk_decomposition(weights, cov_mat.values, tickers)

        return PortfolioResult(
            method="Equal Weighted (1/N)",
            weights=pd.Series(weights, index=tickers),
            expected_return_annual=round(exp_ret, 4),
            volatility_annual=round(port_vol, 4),
            sharpe_ratio=round(sharpe, 2),
            sortino_ratio=sortino,
            cvar_95=cvar,
            marginal_risk_contributions=mcr.round(4),
            percentage_risk_contributions=pcr.round(2),
            diversification_ratio=div_ratio,
        )

    # 2. Maximum Sharpe Portfolio (Markowitz)
    def max_sharpe(
        self,
        returns_df: pd.DataFrame,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
    ) -> PortfolioResult:
        n = returns_df.shape[1]
        tickers = list(returns_df.columns)
        mean_ret, cov_mat = self._annualized_stats(returns_df)
        cov_values = cov_mat.values
        mean_values = mean_ret.values

        def neg_sharpe(w):
            r = np.dot(w, mean_values)
            vol = np.sqrt(np.dot(w.T, np.dot(cov_values, w)))
            return -(r - self.rf) / (vol + 1e-12)

        init_w = np.ones(n) / n
        bounds = [(min_weight, max_weight) for _ in range(n)]
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        res = minimize(neg_sharpe, init_w, method="SLSQP", bounds=bounds, constraints=constraints)
        weights = np.clip(res.x, 0.0, 1.0)
        weights = weights / np.sum(weights)

        exp_ret = float(np.dot(weights, mean_ret))
        port_vol = float(np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights))))
        sharpe = (exp_ret - self.rf) / (port_vol + 1e-12)
        cvar = self.calculate_cvar(returns_df, weights)
        sortino = self.calculate_sortino(returns_df, weights)
        mcr, pcr, div_ratio = self.risk_decomposition(weights, cov_values, tickers)

        return PortfolioResult(
            method="Maximum Sharpe (Markowitz)",
            weights=pd.Series(weights, index=tickers),
            expected_return_annual=round(exp_ret, 4),
            volatility_annual=round(port_vol, 4),
            sharpe_ratio=round(sharpe, 2),
            sortino_ratio=sortino,
            cvar_95=cvar,
            marginal_risk_contributions=mcr.round(4),
            percentage_risk_contributions=pcr.round(2),
            diversification_ratio=div_ratio,
        )

    # 3. Minimum Variance Portfolio
    def minimum_variance(
        self,
        returns_df: pd.DataFrame,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
    ) -> PortfolioResult:
        n = returns_df.shape[1]
        tickers = list(returns_df.columns)
        mean_ret, cov_mat = self._annualized_stats(returns_df)
        cov_values = cov_mat.values

        def port_variance(w):
            return np.dot(w.T, np.dot(cov_values, w))

        init_w = np.ones(n) / n
        bounds = [(min_weight, max_weight) for _ in range(n)]
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        res = minimize(port_variance, init_w, method="SLSQP", bounds=bounds, constraints=constraints)
        weights = np.clip(res.x, 0.0, 1.0)
        weights = weights / np.sum(weights)

        exp_ret = float(np.dot(weights, mean_ret))
        port_vol = float(np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights))))
        sharpe = (exp_ret - self.rf) / (port_vol + 1e-12)
        cvar = self.calculate_cvar(returns_df, weights)
        sortino = self.calculate_sortino(returns_df, weights)
        mcr, pcr, div_ratio = self.risk_decomposition(weights, cov_values, tickers)

        return PortfolioResult(
            method="Minimum Variance",
            weights=pd.Series(weights, index=tickers),
            expected_return_annual=round(exp_ret, 4),
            volatility_annual=round(port_vol, 4),
            sharpe_ratio=round(sharpe, 2),
            sortino_ratio=sortino,
            cvar_95=cvar,
            marginal_risk_contributions=mcr.round(4),
            percentage_risk_contributions=pcr.round(2),
            diversification_ratio=div_ratio,
        )

    # 4. Risk Parity (Equal Risk Contribution)
    def risk_parity(
        self,
        returns_df: pd.DataFrame,
        min_weight: float = 0.01,
        max_weight: float = 0.80,
    ) -> PortfolioResult:
        n = returns_df.shape[1]
        tickers = list(returns_df.columns)
        mean_ret, cov_mat = self._annualized_stats(returns_df)
        cov_values = cov_mat.values

        # Target risk contribution per asset is 1/N
        target_rc = 1.0 / n

        def risk_budget_objective(w):
            port_var = np.dot(w.T, np.dot(cov_values, w))
            mcr = np.dot(cov_values, w)
            rc = (w * mcr) / (port_var + 1e-12)
            return np.sum((rc - target_rc) ** 2)

        init_w = np.ones(n) / n
        bounds = [(min_weight, max_weight) for _ in range(n)]
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        res = minimize(risk_budget_objective, init_w, method="SLSQP", bounds=bounds, constraints=constraints, tol=1e-8)
        weights = np.clip(res.x, 0.0, 1.0)
        weights = weights / np.sum(weights)

        exp_ret = float(np.dot(weights, mean_ret))
        port_vol = float(np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights))))
        sharpe = (exp_ret - self.rf) / (port_vol + 1e-12)
        cvar = self.calculate_cvar(returns_df, weights)
        sortino = self.calculate_sortino(returns_df, weights)
        mcr, pcr, div_ratio = self.risk_decomposition(weights, cov_values, tickers)

        return PortfolioResult(
            method="Risk Parity (ERC)",
            weights=pd.Series(weights, index=tickers),
            expected_return_annual=round(exp_ret, 4),
            volatility_annual=round(port_vol, 4),
            sharpe_ratio=round(sharpe, 2),
            sortino_ratio=sortino,
            cvar_95=cvar,
            marginal_risk_contributions=mcr.round(4),
            percentage_risk_contributions=pcr.round(2),
            diversification_ratio=div_ratio,
        )

    # 5. Black-Litterman Portfolio Model
    def black_litterman(
        self,
        returns_df: pd.DataFrame,
        market_caps: Optional[Dict[str, float]] = None,
        views: Optional[Dict[str, float]] = None,  # Absolute view: {ticker: expected_return}
        view_confidences: Optional[Dict[str, float]] = None,  # e.g. {ticker: 0.5}
        tau: float = 0.05,
    ) -> PortfolioResult:
        """
        Black-Litterman model with market equilibrium prior combined with subjective/model views.
        """
        n = returns_df.shape[1]
        tickers = list(returns_df.columns)
        mean_ret, cov_mat = self._annualized_stats(returns_df)
        sigma = cov_mat.values

        # Market Capitalization Weights Prior (default equal weight if none provided)
        if market_caps:
            total_cap = sum(market_caps.get(t, 1.0) for t in tickers)
            w_mkt = np.array([market_caps.get(t, 1.0) / total_cap for t in tickers])
        else:
            w_mkt = np.ones(n) / n

        # Risk aversion coefficient lambda = (r_m - r_f) / var_m
        var_mkt = float(np.dot(w_mkt.T, np.dot(sigma, w_mkt)))
        r_mkt = float(np.dot(w_mkt, mean_ret))
        risk_aversion = max(1.0, (r_mkt - self.rf) / (var_mkt + 1e-12))

        # Implied Equilibrium Excess Returns: Pi = lambda * Sigma * w_mkt
        pi = risk_aversion * np.dot(sigma, w_mkt)

        # In case no investor views specified, posterior equals equilibrium
        if not views:
            mu_bl = pi + self.rf
            cov_bl = sigma * (1.0 + tau)
        else:
            # Construct P, Q, Omega
            k = len(views)
            P = np.zeros((k, n))
            Q = np.zeros(k)
            omega_diag = []

            for i, (ticker, view_ret) in enumerate(views.items()):
                if ticker in tickers:
                    idx = tickers.index(ticker)
                    P[i, idx] = 1.0
                    Q[i] = view_ret - self.rf
                    conf = view_confidences.get(ticker, 0.5) if view_confidences else 0.5
                    # Omega proportional to view variance: tau * P * Sigma * P^T / conf
                    v_var = float(tau * np.dot(P[i], np.dot(sigma, P[i].T)) / (conf + 1e-6))
                    omega_diag.append(max(1e-6, v_var))

            Omega = np.diag(omega_diag)

            # Master Formula for Posterior Returns:
            # E[R] = [ (tau*Sigma)^-1 + P^T * Omega^-1 * P ]^-1 * [ (tau*Sigma)^-1 * Pi + P^T * Omega^-1 * Q ]
            tau_sigma_inv = np.linalg.inv(tau * sigma)
            omega_inv = np.linalg.inv(Omega)

            m_inv = tau_sigma_inv + np.dot(P.T, np.dot(omega_inv, P))
            m = np.linalg.inv(m_inv)

            posterior_excess = np.dot(m, (np.dot(tau_sigma_inv, pi) + np.dot(P.T, np.dot(omega_inv, Q))))
            mu_bl = posterior_excess + self.rf
            cov_bl = sigma + m

        # Solve constrained weights on posterior parameters
        def obj_func(w):
            r = np.dot(w, mu_bl)
            vol = np.sqrt(np.dot(w.T, np.dot(cov_bl, w)))
            return -(r - self.rf) / (vol + 1e-12)

        init_w = np.ones(n) / n
        bounds = [(0.0, 1.0) for _ in range(n)]
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        res = minimize(obj_func, init_w, method="SLSQP", bounds=bounds, constraints=constraints)
        weights = np.clip(res.x, 0.0, 1.0)
        weights = weights / np.sum(weights)

        exp_ret = float(np.dot(weights, mu_bl))
        port_vol = float(np.sqrt(np.dot(weights.T, np.dot(cov_bl, weights))))
        sharpe = (exp_ret - self.rf) / (port_vol + 1e-12)
        cvar = self.calculate_cvar(returns_df, weights)
        sortino = self.calculate_sortino(returns_df, weights)
        mcr, pcr, div_ratio = self.risk_decomposition(weights, cov_bl, tickers)

        return PortfolioResult(
            method="Black-Litterman Posterior",
            weights=pd.Series(weights, index=tickers),
            expected_return_annual=round(exp_ret, 4),
            volatility_annual=round(port_vol, 4),
            sharpe_ratio=round(sharpe, 2),
            sortino_ratio=sortino,
            cvar_95=cvar,
            marginal_risk_contributions=mcr.round(4),
            percentage_risk_contributions=pcr.round(2),
            diversification_ratio=div_ratio,
        )

    # 6. CVaR (Expected Shortfall) Optimization
    def min_cvar(
        self,
        returns_df: pd.DataFrame,
        alpha: float = 0.95,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
    ) -> PortfolioResult:
        n = returns_df.shape[1]
        tickers = list(returns_df.columns)
        mean_ret, cov_mat = self._annualized_stats(returns_df)
        R = returns_df.values  # (T, n)
        T_obs = len(R)

        # Variables: w (n,), gamma (1,)
        # Minimize gamma + 1/(T * (1-alpha)) * sum( max(0, -R*w - gamma) )
        def cvar_objective(params):
            w = params[:n]
            gamma = params[n]
            port_loss = -np.dot(R, w)
            excess_loss = np.maximum(0.0, port_loss - gamma)
            return gamma + (1.0 / (T_obs * (1.0 - alpha))) * np.sum(excess_loss)

        init_params = np.append(np.ones(n) / n, 0.02)
        bounds = [(min_weight, max_weight) for _ in range(n)] + [(None, None)]
        constraints = [{"type": "eq", "fun": lambda p: np.sum(p[:n]) - 1.0}]

        res = minimize(cvar_objective, init_params, method="SLSQP", bounds=bounds, constraints=constraints)
        weights = np.clip(res.x[:n], 0.0, 1.0)
        weights = weights / np.sum(weights)

        exp_ret = float(np.dot(weights, mean_ret))
        port_vol = float(np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights))))
        sharpe = (exp_ret - self.rf) / (port_vol + 1e-12)
        cvar = self.calculate_cvar(returns_df, weights, alpha=alpha)
        sortino = self.calculate_sortino(returns_df, weights)
        mcr, pcr, div_ratio = self.risk_decomposition(weights, cov_mat.values, tickers)

        return PortfolioResult(
            method=f"Min CVaR ({int(alpha*100)}%)",
            weights=pd.Series(weights, index=tickers),
            expected_return_annual=round(exp_ret, 4),
            volatility_annual=round(port_vol, 4),
            sharpe_ratio=round(sharpe, 2),
            sortino_ratio=sortino,
            cvar_95=cvar,
            marginal_risk_contributions=mcr.round(4),
            percentage_risk_contributions=pcr.round(2),
            diversification_ratio=div_ratio,
        )

    # 7. Trace Efficient Frontier
    def compute_efficient_frontier(
        self, returns_df: pd.DataFrame, num_points: int = 30
    ) -> EfficientFrontierResult:
        n = returns_df.shape[1]
        tickers = list(returns_df.columns)
        mean_ret, cov_mat = self._annualized_stats(returns_df)
        cov_values = cov_mat.values
        mean_values = mean_ret.values

        min_r = float(mean_values.min())
        max_r = float(mean_values.max())
        targets = np.linspace(min_r, max_r, num_points)

        vols = []
        weights_list = []
        sharpes = []

        init_w = np.ones(n) / n
        bounds = [(0.0, 1.0) for _ in range(n)]

        for target in targets:
            constraints = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
                {"type": "eq", "fun": lambda w, t=target: np.dot(w, mean_values) - t},
            ]
            res = minimize(
                lambda w: np.dot(w.T, np.dot(cov_values, w)),
                init_w,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
            )
            if res.success:
                w = res.x / np.sum(res.x)
                vol = float(np.sqrt(np.dot(w.T, np.dot(cov_values, w))))
                sh = (target - self.rf) / (vol + 1e-12)
                vols.append(vol)
                weights_list.append(w)
                sharpes.append(sh)
            else:
                # If solver doesn't find exact match, record NaN
                vols.append(np.nan)
                weights_list.append(np.zeros(n))
                sharpes.append(np.nan)

        return EfficientFrontierResult(
            target_returns=targets,
            target_volatilities=np.array(vols),
            weights_matrix=np.array(weights_list),
            sharpe_ratios=np.array(sharpes),
            tickers=tickers,
        )
