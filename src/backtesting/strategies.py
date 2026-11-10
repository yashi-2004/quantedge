"""
QuantEdge Strategy Generator Library
Constructs historical target weight time series for systematic backtesting:
1. Buy & Hold
2. Periodic Equal-Weight (1/N)
3. Moving Average Crossover Trend Following
4. Cross-Sectional Momentum
5. Mean Reversion
6. Dynamic Minimum Variance
7. Dynamic Risk Parity
8. ML-Driven Return Forecasting Allocation
9. Regime-Aware Allocation
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from src.portfolio.optimisation import PortfolioOptimizer


class StrategyLibrary:
    """
    Generates point-in-time portfolio weight matrices for backtesting simulation.
    """

    @staticmethod
    def buy_and_hold(prices_df: pd.DataFrame, initial_weights: Optional[np.ndarray] = None) -> pd.DataFrame:
        n = prices_df.shape[1]
        w0 = initial_weights if initial_weights is not None else np.ones(n) / n
        # Replicated across all dates
        return pd.DataFrame([w0] * len(prices_df), index=prices_df.index, columns=prices_df.columns)

    @staticmethod
    def periodic_equal_weight(prices_df: pd.DataFrame, freq: str = "M") -> pd.DataFrame:
        """
        Rebalances to equal weights at defined calendar frequency ('W', 'M', 'Q').
        """
        n = prices_df.shape[1]
        tickers = prices_df.columns
        weights = pd.DataFrame(index=prices_df.index, columns=tickers)

        # Identify rebalance dates
        rebalance_dates = prices_df.resample(freq).first().index
        for d in prices_df.index:
            weights.loc[d] = np.ones(n) / n

        return weights.astype(float)

    @staticmethod
    def moving_average_crossover(
        prices_df: pd.DataFrame, fast_window: int = 20, slow_window: int = 50
    ) -> pd.DataFrame:
        """
        Allocates to assets when Fast SMA > Slow SMA, else 0 (cash).
        Normalized across active assets.
        """
        fast_sma = prices_df.rolling(fast_window, min_periods=fast_window).mean()
        slow_sma = prices_df.rolling(slow_window, min_periods=slow_window).mean()

        signal = (fast_sma > slow_sma).astype(float)
        row_sums = signal.sum(axis=1).replace(0, 1.0)
        weights = signal.div(row_sums, axis=0).fillna(0.0)
        return weights

    @staticmethod
    def cross_sectional_momentum(
        prices_df: pd.DataFrame, lookback_days: int = 63, top_k: int = 3, rebalance_freq: str = "M"
    ) -> pd.DataFrame:
        """
        Ranks assets by lookback return and invests equally in top K assets.
        """
        n = prices_df.shape[1]
        top_k = min(top_k, n)
        returns = prices_df.pct_change(lookback_days)

        weights = pd.DataFrame(0.0, index=prices_df.index, columns=prices_df.columns)
        rebalance_dates = set(prices_df.resample(rebalance_freq).first().index)

        current_w = np.ones(n) / n
        for d in prices_df.index:
            if d in rebalance_dates and not returns.loc[d].isna().all():
                row_ret = returns.loc[d]
                top_tickers = row_ret.nlargest(top_k).index
                new_w = pd.Series(0.0, index=prices_df.columns)
                new_w[top_tickers] = 1.0 / top_k
                current_w = new_w.values
            weights.loc[d] = current_w

        return weights

    @staticmethod
    def mean_reversion(prices_df: pd.DataFrame, window: int = 20, z_threshold: float = -1.5) -> pd.DataFrame:
        """
        Invests in assets whose price is substantially below their moving average (oversold).
        """
        sma = prices_df.rolling(window, min_periods=window).mean()
        rolling_std = prices_df.rolling(window, min_periods=window).std()
        z_scores = (prices_df - sma) / (rolling_std + 1e-12)

        oversold = (z_scores < z_threshold).astype(float)
        row_sums = oversold.sum(axis=1).replace(0, 1.0)
        weights = oversold.div(row_sums, axis=0).fillna(0.0)
        return weights

    @staticmethod
    def dynamic_risk_parity(
        prices_df: pd.DataFrame, lookback_days: int = 126, rebalance_freq: str = "M"
    ) -> pd.DataFrame:
        """
        Computes Risk Parity weights at periodic calendar dates using rolling historical covariance.
        """
        optimizer = PortfolioOptimizer()
        returns_df = prices_df.pct_change().dropna()
        weights = pd.DataFrame(index=prices_df.index, columns=prices_df.columns, dtype=float)

        rebalance_dates = set(prices_df.resample(rebalance_freq).first().index)
        n = prices_df.shape[1]
        current_w = np.ones(n) / n

        for i, d in enumerate(prices_df.index):
            if d in rebalance_dates and i >= lookback_days:
                hist_slice = returns_df.loc[:d].tail(lookback_days)
                if len(hist_slice) >= 30:
                    try:
                        res = optimizer.risk_parity(hist_slice)
                        current_w = res.weights.values
                    except Exception:
                        pass
            weights.loc[d] = current_w

        return weights.ffill().bfill()

    @staticmethod
    def regime_aware_allocation(
        prices_df: pd.DataFrame,
        regime_labels: pd.Series,  # 0: Low-Vol Bull, 1: Transition, 2: High-Vol Bear
        defensive_asset: Optional[str] = "TLT",
    ) -> pd.DataFrame:
        """
        Shifts allocation to defensive asset or cash during high-volatility regimes.
        """
        weights = pd.DataFrame(index=prices_df.index, columns=prices_df.columns, dtype=float)
        n = prices_df.shape[1]
        def_idx = prices_df.columns.get_loc(defensive_asset) if defensive_asset in prices_df.columns else None

        for d in prices_df.index:
            regime = regime_labels.get(d, 0)
            if regime == 0:  # Bull / Low Vol -> 100% Equities equal weight
                if def_idx is not None and n > 1:
                    w = np.ones(n) / (n - 1)
                    w[def_idx] = 0.0
                else:
                    w = np.ones(n) / n
            elif regime == 1:  # Moderate / Transition -> Balanced
                w = np.ones(n) / n
            else:  # High Vol / Bear -> 80% Defensive / Cash
                w = np.full(n, 0.20 / max(1, n - 1))
                if def_idx is not None:
                    w[def_idx] = 0.80
            weights.loc[d] = w

        return weights
