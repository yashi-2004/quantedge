"""
QuantEdge Technical Indicators Engine
Calculates standard and advanced technical indicators strictly using historical information.
Zero forward-looking bias.
"""

import numpy as np
import pandas as pd


def calculate_sma(series: pd.Series, window: int = 20) -> pd.Series:
    """Simple Moving Average"""
    return series.rolling(window=window, min_periods=window).mean()


def calculate_ema(series: pd.Series, span: int = 20) -> pd.Series:
    """Exponential Moving Average"""
    return series.ewm(span=span, adjust=False).mean()


def calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Wilder's Smoothed Relative Strength Index (RSI).
    Formula: RSI = 100 - (100 / (1 + RS))
    where RS = AvgGain / AvgLoss
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-12)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def calculate_macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """
    Moving Average Convergence Divergence (MACD).
    MACD Line = Fast EMA - Slow EMA
    Signal Line = 9-period EMA of MACD Line
    Histogram = MACD Line - Signal Line
    """
    ema_fast = calculate_ema(series, span=fast)
    ema_slow = calculate_ema(series, span=slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, span=signal)
    hist = macd_line - signal_line

    return pd.DataFrame(
        {
            f"MACD_{fast}_{slow}": macd_line,
            f"MACD_Signal_{signal}": signal_line,
            "MACD_Hist": hist,
        },
        index=series.index,
    )


def calculate_bollinger_bands(
    series: pd.Series, window: int = 20, num_std: float = 2.0
) -> pd.DataFrame:
    """
    Bollinger Bands: Middle (SMA), Upper (SMA + k*sigma), Lower (SMA - k*sigma),
    %B: (Price - Lower) / (Upper - Lower),
    Bandwidth: (Upper - Lower) / Middle
    """
    mid = calculate_sma(series, window=window)
    rolling_std = series.rolling(window=window, min_periods=window).std()
    upper = mid + num_std * rolling_std
    lower = mid - num_std * rolling_std
    pct_b = (series - lower) / (upper - lower + 1e-12)
    bandwidth = (upper - lower) / (mid + 1e-12)

    return pd.DataFrame(
        {
            f"BB_Mid_{window}": mid,
            f"BB_Upper_{window}": upper,
            f"BB_Lower_{window}": lower,
            f"BB_PctB_{window}": pct_b,
            f"BB_Bandwidth_{window}": bandwidth,
        },
        index=series.index,
    )


def calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Average True Range (ATR).
    TR = max(High - Low, |High - Close_prev|, |Low - Close_prev|)
    """
    prev_close = df["Close"].shift(1)
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - prev_close).abs()
    tr3 = (df["Low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    return atr


def calculate_stochastic(
    df: pd.DataFrame, k_window: int = 14, d_window: int = 3
) -> pd.DataFrame:
    """
    Stochastic Oscillator:
    %K = (Close - Lowest Low) / (Highest High - Lowest Low) * 100
    %D = 3-period SMA of %K
    """
    lowest_low = df["Low"].rolling(window=k_window, min_periods=k_window).min()
    highest_high = df["High"].rolling(window=k_window, min_periods=k_window).max()
    k_percent = 100.0 * (df["Close"] - lowest_low) / (highest_high - lowest_low + 1e-12)
    d_percent = k_percent.rolling(window=d_window, min_periods=d_window).mean()

    return pd.DataFrame(
        {f"Stoch_K_{k_window}": k_percent, f"Stoch_D_{d_window}": d_percent},
        index=df.index,
    )


def calculate_adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Average Directional Index (ADX) measures trend strength regardless of direction.
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    prev_high = high.shift(1)
    prev_low = low.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    alpha = 1.0 / window
    atr = tr.ewm(alpha=alpha, min_periods=window, adjust=False).mean()
    plus_di = 100.0 * pd.Series(plus_dm, index=df.index).ewm(alpha=alpha, min_periods=window, adjust=False).mean() / (atr + 1e-12)
    minus_di = 100.0 * pd.Series(minus_dm, index=df.index).ewm(alpha=alpha, min_periods=window, adjust=False).mean() / (atr + 1e-12)

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12)
    adx = dx.ewm(alpha=alpha, min_periods=window, adjust=False).mean()
    return adx


def calculate_obv(df: pd.DataFrame) -> pd.Series:
    """
    On-Balance Volume (OBV):
    Cumulative volume added on up days and subtracted on down days.
    """
    if "Volume" not in df.columns:
        return pd.Series(0.0, index=df.index)
    close_diff = df["Close"].diff()
    direction = np.sign(close_diff).fillna(0)
    obv = (direction * df["Volume"]).cumsum()
    return obv
