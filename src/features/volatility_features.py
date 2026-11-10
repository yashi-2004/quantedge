"""
QuantEdge Advanced Volatility Estimators Engine
Implements research-grade realized, range-based, and jump-robust volatility estimators:
- Close-to-Close Realized Volatility
- Parkinson (1980) High-Low Estimator
- Garman-Klass (1980) OHLC Estimator
- Rogers-Satchell (1991) Non-Zero Drift Estimator
- Yang-Zhang (2000) Overnight Jump-Robust Estimator
- Volatility of Volatility (Vol-of-Vol)
"""

from typing import List
import numpy as np
import pandas as pd


def calculate_realized_volatility(returns: pd.Series, window: int = 21, annualize: bool = True) -> pd.Series:
    """
    Standard Close-to-Close Realized Volatility.
    """
    factor = np.sqrt(252) if annualize else 1.0
    return returns.rolling(window=window, min_periods=window).std() * factor


def calculate_parkinson_volatility(df: pd.DataFrame, window: int = 21, annualize: bool = True) -> pd.Series:
    """
    Parkinson (1980) High-Low Volatility Estimator.
    5x more efficient than close-to-close volatility under geometric Brownian motion.
    sigma_P = sqrt( 1/(4*ln(2)*N) * sum(ln(H/L)^2) ) * sqrt(252)
    """
    log_hl = np.log(df["High"] / df["Low"])
    squared_hl = log_hl**2
    factor = (252.0 if annualize else 1.0) / (4.0 * np.log(2.0))
    rolling_var = squared_hl.rolling(window=window, min_periods=window).mean() * factor
    return np.sqrt(rolling_var)


def calculate_garman_klass_volatility(df: pd.DataFrame, window: int = 21, annualize: bool = True) -> pd.Series:
    """
    Garman-Klass (1980) OHLC Volatility Estimator.
    Up to 8x more efficient than close-to-close volatility.
    term1 = 0.5 * (ln(H/L))^2
    term2 = (2*ln(2) - 1) * (ln(C/O))^2
    """
    log_hl = np.log(df["High"] / df["Low"])
    log_co = np.log(df["Close"] / df["Open"])

    gk_component = 0.5 * (log_hl**2) - (2.0 * np.log(2.0) - 1.0) * (log_co**2)
    factor = 252.0 if annualize else 1.0
    rolling_var = gk_component.rolling(window=window, min_periods=window).mean() * factor
    return np.sqrt(np.maximum(0, rolling_var))


def calculate_rogers_satchell_volatility(df: pd.DataFrame, window: int = 21, annualize: bool = True) -> pd.Series:
    """
    Rogers-Satchell (1991) Volatility Estimator.
    Robust to non-zero drift.
    sigma_RS^2 = sum[ ln(H/C)*ln(H/O) + ln(L/C)*ln(L/O) ] / N
    """
    log_hc = np.log(df["High"] / df["Close"])
    log_ho = np.log(df["High"] / df["Open"])
    log_lc = np.log(df["Low"] / df["Close"])
    log_lo = np.log(df["Low"] / df["Open"])

    rs_component = (log_hc * log_ho) + (log_lc * log_lo)
    factor = 252.0 if annualize else 1.0
    rolling_var = rs_component.rolling(window=window, min_periods=window).mean() * factor
    return np.sqrt(np.maximum(0, rolling_var))


def calculate_yang_zhang_volatility(df: pd.DataFrame, window: int = 21, annualize: bool = True) -> pd.Series:
    """
    Yang-Zhang (2000) Volatility Estimator.
    Minimum variance unbiased estimator robust to opening jump gaps and non-zero drift.
    """
    log_co_prev = np.log(df["Open"] / df["Close"].shift(1))  # overnight jump
    log_cc = np.log(df["Close"] / df["Close"].shift(1))  # close-to-close
    log_ho = np.log(df["High"] / df["Open"])
    log_lo = np.log(df["Low"] / df["Open"])
    log_co = np.log(df["Close"] / df["Open"])

    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    factor = 252.0 if annualize else 1.0

    # Overnight variance
    var_overnight = log_co_prev.rolling(window=window, min_periods=window).var()
    # Open-to-close variance
    var_open_to_close = log_co.rolling(window=window, min_periods=window).var()
    # Rogers-Satchell variance
    rs_comp = (log_ho * (log_ho - log_co)) + (log_lo * (log_lo - log_co))
    var_rs = rs_comp.rolling(window=window, min_periods=window).mean()

    yz_var = (var_overnight + k * var_open_to_close + (1 - k) * var_rs) * factor
    return np.sqrt(np.maximum(0, yz_var))


def calculate_volatility_of_volatility(vol_series: pd.Series, window: int = 21) -> pd.Series:
    """
    Volatility of Volatility (Vol-of-Vol):
    Rolling standard deviation of a chosen volatility time series.
    """
    return vol_series.rolling(window=window, min_periods=window).std()
