"""
QuantEdge Realistic Backtesting & Historical Simulation Engine
Implements realistic market microstructure assumptions:
- Transaction costs & linear slippage models
- Execution delay (signal at t, execution at t+1) to prevent look-ahead bias
- Periodic rebalancing (Daily, Weekly, Monthly, Quarterly)
- Comprehensive risk-adjusted metrics: CAGR, Sharpe, Sortino, Calmar, Max Drawdown, Turnover
- Benchmark comparison (Alpha, Beta, Information Ratio)
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    strategy_name: str
    equity_curve: pd.Series
    daily_returns: pd.Series
    drawdown_curve: pd.Series
    positions_history: pd.DataFrame  # Weights over time
    turnover_series: pd.Series
    benchmark_equity_curve: Optional[pd.Series]
    cagr: float
    total_return: float
    annual_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration_days: int
    win_rate: float
    profit_factor: float
    total_turnover: float
    total_costs_paid: float
    alpha: Optional[float]
    beta: Optional[float]
    information_ratio: Optional[float]
    monthly_returns_table: pd.DataFrame


class BacktestEngine:
    """
    Simulates portfolio execution with execution friction, slippage, and temporal delays.
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        transaction_cost_bps: float = 10.0,  # 10 bps = 0.0010
        slippage_bps: float = 5.0,  # 5 bps = 0.0005
        risk_free_rate: float = 0.03,
        execution_lag: int = 1,  # 1 day lag: signal at t executed at t+1
    ):
        self.initial_capital = initial_capital
        self.cost_rate = (transaction_cost_bps + slippage_bps) / 10_000.0
        self.rf = risk_free_rate
        self.execution_lag = execution_lag

    def run_backtest(
        self,
        prices_df: pd.DataFrame,
        weights_df: pd.DataFrame,
        strategy_name: str = "Quantitative Strategy",
        benchmark_series: Optional[pd.Series] = None,
    ) -> BacktestResult:
        """
        Executes backtest given asset prices and target portfolio weights.
        Applies execution lag to weights strictly preventing look-ahead bias.
        """
        # Align indexes
        common_idx = prices_df.index.intersection(weights_df.index)
        if len(common_idx) < 10:
            raise ValueError("Insufficient overlapping dates between prices and weights.")

        P = prices_df.loc[common_idx].sort_index()
        W_target = weights_df.loc[common_idx].sort_index()

        # Shift target weights by execution_lag to enforce causality!
        # At time t, the portfolio holds weights decided at t - execution_lag
        if self.execution_lag > 0:
            W_held = W_target.shift(self.execution_lag).fillna(0.0)
        else:
            W_held = W_target.copy()

        # Asset returns
        asset_returns = P.pct_change().fillna(0.0)

        n_steps = len(common_idx)
        portfolio_values = np.zeros(n_steps)
        portfolio_values[0] = self.initial_capital
        daily_turnover = np.zeros(n_steps)
        costs_paid = np.zeros(n_steps)
        net_returns = np.zeros(n_steps)

        current_weights = W_held.iloc[0].values

        for t in range(1, n_steps):
            target_w = W_held.iloc[t].values

            # Turnover = sum |target_w - prior_w|
            turnover = float(np.sum(np.abs(target_w - current_weights)))
            cost = portfolio_values[t - 1] * turnover * self.cost_rate
            daily_turnover[t] = turnover
            costs_paid[t] = cost

            # Gross portfolio return from held weights
            r_t = float(np.dot(target_w, asset_returns.iloc[t].values))

            # Portfolio value after return and transaction cost
            val_after_return = portfolio_values[t - 1] * (1.0 + r_t) - cost
            portfolio_values[t] = max(0.0, val_after_return)
            net_returns[t] = (portfolio_values[t] - portfolio_values[t - 1]) / (portfolio_values[t - 1] + 1e-12)

            # Update current weights drifted by price returns
            denom = 1.0 + r_t
            if denom > 0:
                current_weights = (target_w * (1.0 + asset_returns.iloc[t].values)) / denom
            else:
                current_weights = target_w

        equity_s = pd.Series(portfolio_values, index=common_idx, name="Portfolio_Equity")
        daily_ret_s = pd.Series(net_returns, index=common_idx, name="Net_Returns")
        turnover_s = pd.Series(daily_turnover, index=common_idx, name="Turnover")

        # Drawdown calculation
        rolling_max = equity_s.cummax()
        drawdown_s = (equity_s - rolling_max) / (rolling_max + 1e-12)

        # Performance Metrics
        total_ret = float((portfolio_values[-1] / portfolio_values[0]) - 1.0)
        n_years = max(0.1, len(common_idx) / 252.0)
        cagr = float(((portfolio_values[-1] / portfolio_values[0]) ** (1.0 / n_years)) - 1.0)
        ann_vol = float(daily_ret_s.std() * np.sqrt(252))
        sharpe = float((cagr - self.rf) / (ann_vol + 1e-12))

        # Downside deviation & Sortino
        downside_ret = daily_ret_s[daily_ret_s < (self.rf / 252.0)]
        downside_std = float(downside_ret.std() * np.sqrt(252)) if len(downside_ret) > 1 else 1e-6
        sortino = float((cagr - self.rf) / (downside_std + 1e-12))

        max_dd = float(drawdown_s.min())
        calmar = float(cagr / (abs(max_dd) + 1e-12)) if max_dd < 0 else 0.0

        # Drawdown duration in days
        is_in_dd = drawdown_s < 0
        dd_durations = (~is_in_dd).cumsum()[is_in_dd].value_counts()
        max_dd_duration = int(dd_durations.max()) if not dd_durations.empty else 0

        # Win Rate & Profit Factor
        pos_days = daily_ret_s[daily_ret_s > 0]
        neg_days = daily_ret_s[daily_ret_s < 0]
        win_rate = float(len(pos_days) / max(1, (len(pos_days) + len(neg_days))))
        gross_profit = float(pos_days.sum())
        gross_loss = float(abs(neg_days.sum()))
        profit_factor = float(gross_profit / (gross_loss + 1e-12)) if gross_loss > 0 else float("inf")

        total_turnover = float(turnover_s.sum())
        total_costs = float(np.sum(costs_paid))

        # Benchmark Comparison if available
        alpha_val = None
        beta_val = None
        info_ratio = None
        bench_equity_s = None

        if benchmark_series is not None and not benchmark_series.empty:
            bench_aligned = benchmark_series.loc[benchmark_series.index.intersection(common_idx)]
            if len(bench_aligned) > 10:
                bench_ret = bench_aligned.pct_change().fillna(0.0)
                bench_equity_s = (1.0 + bench_ret).cumprod() * self.initial_capital

                cov = float(np.cov(daily_ret_s.values, bench_ret.values)[0, 1])
                bench_var = float(np.var(bench_ret.values))
                beta_val = round(cov / (bench_var + 1e-12), 3)
                bench_cagr = float(((bench_equity_s.iloc[-1] / bench_equity_s.iloc[0]) ** (1.0 / n_years)) - 1.0)
                alpha_val = round((cagr - (self.rf + beta_val * (bench_cagr - self.rf))) * 100.0, 2)

                tracking_diff = daily_ret_s - bench_ret
                tracking_error = float(tracking_diff.std() * np.sqrt(252))
                info_ratio = round(float((cagr - bench_cagr) / (tracking_error + 1e-12)), 2)

        # Monthly returns matrix table
        daily_ret_df = daily_ret_s.to_frame("ret")
        daily_ret_df["Year"] = daily_ret_df.index.year
        daily_ret_df["Month"] = daily_ret_df.index.month
        monthly_table = (
            daily_ret_df.groupby(["Year", "Month"])["ret"]
            .apply(lambda x: (1.0 + x).prod() - 1.0)
            .unstack(level=1)
            * 100.0
        ).round(2)

        return BacktestResult(
            strategy_name=strategy_name,
            equity_curve=equity_s,
            daily_returns=daily_ret_s,
            drawdown_curve=drawdown_s * 100.0,
            positions_history=W_held,
            turnover_series=turnover_s,
            benchmark_equity_curve=bench_equity_s,
            cagr=round(cagr * 100.0, 2),
            total_return=round(total_ret * 100.0, 2),
            annual_volatility=round(ann_vol * 100.0, 2),
            sharpe_ratio=round(sharpe, 2),
            sortino_ratio=round(sortino, 2),
            calmar_ratio=round(calmar, 2),
            max_drawdown=round(max_dd * 100.0, 2),
            max_drawdown_duration_days=max_dd_duration,
            win_rate=round(win_rate * 100.0, 1),
            profit_factor=round(min(99.9, profit_factor), 2),
            total_turnover=round(total_turnover, 2),
            total_costs_paid=round(total_costs, 2),
            alpha=alpha_val,
            beta=beta_val,
            information_ratio=info_ratio,
            monthly_returns_table=monthly_table,
        )
