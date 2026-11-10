"""
QuantEdge Statistical Features Engine
Extracts rigorous statistical metrics and return dynamics for time series modeling.
Ensures zero forward-looking leakage.
"""

from typing import List, Optional
import numpy as np
import pandas as pd


def calculate_returns(df: pd.DataFrame, col: str = "Close") -> pd.DataFrame:
    """
    Computes simple returns, log returns, and overnight gaps.
    """
    price = df[col].astype(float)
    simple_ret = price.pct_change()
    log_ret = np.log(price / price.shift(1))

    # Overnight gap: (Open_t - Close_{t-1}) / Close_{t-1}
    overnight_gap = (df["Open"] - price.shift(1)) / price.shift(1) if "Open" in df.columns else pd.Series(0.0, index=df.index)

    # Intraday range: (High_t - Low_t) / Open_t
    intraday_range = (df["High"] - df["Low"]) / (df["Open"] + 1e-12) if "High" in df.columns and "Low" in df.columns and "Open" in df.columns else pd.Series(0.0, index=df.index)

    return pd.DataFrame(
        {
            "Return_Simple": simple_ret,
            "Return_Log": log_ret,
            "Overnight_Gap": overnight_gap,
            "Intraday_Range": intraday_range,
        },
        index=df.index,
    )


def calculate_lagged_features(series: pd.Series, lags: List[int] = [1, 2, 3, 5, 10, 21], prefix: str = "Lag") -> pd.DataFrame:
    """
    Creates strictly backward-looking lagged features: x_{t-k}
    """
    lag_dict = {}
    for lag in lags:
        lag_dict[f"{prefix}_{lag}"] = series.shift(lag)
    return pd.DataFrame(lag_dict, index=series.index)


def calculate_rolling_statistics(returns: pd.Series, windows: List[int] = [21, 63]) -> pd.DataFrame:
    """
    Computes rolling mean, volatility, skewness, and kurtosis.
    """
    feats = {}
    for w in windows:
        roll = returns.rolling(window=w, min_periods=w)
        mean_s = roll.mean()
        std_s = roll.std()
        feats[f"Roll_Mean_{w}"] = mean_s
        feats[f"Roll_Std_{w}"] = std_s
        feats[f"Roll_Skew_{w}"] = roll.skew()
        feats[f"Roll_Kurt_{w}"] = roll.kurt()
        # Annualized Sharpe ratio proxy of returns over window w (assuming 0 risk-free rate)
        feats[f"Roll_Sharpe_{w}"] = (mean_s / (std_s + 1e-12)) * np.sqrt(252)

    return pd.DataFrame(feats, index=returns.index)


def calculate_autocorrelation(returns: pd.Series, window: int = 63, lag: int = 1) -> pd.Series:
    """
    Rolling autocorrelation of returns at lag k:
    corr(r_t, r_{t-k}) over window W.
    """
    shifted = returns.shift(lag)
    return returns.rolling(window=window, min_periods=window).corr(shifted)


def calculate_mean_reversion_score(price: pd.Series, window: int = 20) -> pd.Series:
    """
    Z-score of current price relative to rolling moving average:
    (P_t - SMA_t) / sigma_t
    """
    sma = price.rolling(window=window, min_periods=window).mean()
    rolling_std = price.rolling(window=window, min_periods=window).std()
    return (price - sma) / (rolling_std + 1e-12)


def calculate_rolling_beta_and_correlation(
    asset_returns: pd.Series, benchmark_returns: pd.Series, window: int = 63
) -> pd.DataFrame:
    """
    Calculates rolling market beta and correlation:
    Beta = Cov(r_i, r_m) / Var(r_m)
    """
    combined = pd.concat([asset_returns, benchmark_returns], axis=1).dropna()
    if combined.empty:
        return pd.DataFrame(index=asset_returns.index)

    r_i = combined.iloc[:, 0]
    r_m = combined.iloc[:, 1]

    rolling_cov = r_i.rolling(window=window, min_periods=window).cov(r_m)
    rolling_var_m = r_m.rolling(window=window, min_periods=window).var()
    rolling_corr = r_i.rolling(window=window, min_periods=window).corr(r_m)
    rolling_beta = rolling_cov / (rolling_var_m + 1e-12)

    return pd.DataFrame(
        {f"Rolling_Beta_{window}": rolling_beta, f"Rolling_Corr_{window}": rolling_corr},
        index=combined.index,
    ).reindex(asset_returns.index)
