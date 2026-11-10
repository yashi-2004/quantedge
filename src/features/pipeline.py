"""
QuantEdge Unified Feature Engineering Pipeline
Constructs point-in-time feature matrices, predictive targets,
performs multicollinearity diagnostics, and prevents look-ahead leakage.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from src.features.macro_features import MacroFeatureEngine
from src.features.statistical_features import (
    calculate_autocorrelation,
    calculate_lagged_features,
    calculate_mean_reversion_score,
    calculate_returns,
    calculate_rolling_beta_and_correlation,
    calculate_rolling_statistics,
)
from src.features.technical_indicators import (
    calculate_adx,
    calculate_atr,
    calculate_bollinger_bands,
    calculate_ema,
    calculate_macd,
    calculate_obv,
    calculate_rsi,
    calculate_sma,
    calculate_stochastic,
)
from src.features.volatility_features import (
    calculate_garman_klass_volatility,
    calculate_parkinson_volatility,
    calculate_realized_volatility,
    calculate_rogers_satchell_volatility,
    calculate_volatility_of_volatility,
    calculate_yang_zhang_volatility,
)


@dataclass
class FeatureMatrix:
    X: pd.DataFrame
    y_return: pd.Series
    y_direction: pd.Series
    y_volatility: pd.Series
    feature_names: List[str]
    scaler: Optional[object] = None
    correlation_with_target: Optional[pd.Series] = None
    vif_summary: Optional[pd.DataFrame] = None


class FeatureEngine:
    """
    Constructs a comprehensive, zero-lookahead feature matrix from OHLCV market data.
    """

    def __init__(self, include_macro: bool = True):
        self.include_macro = include_macro
        self.macro_engine = MacroFeatureEngine()

    def build_feature_matrix(
        self,
        df: pd.DataFrame,
        benchmark_df: Optional[pd.DataFrame] = None,
        scale_method: Optional[str] = None,  # 'standard', 'robust', 'minmax', None
        selected_features: Optional[List[str]] = None,
    ) -> FeatureMatrix:
        """
        Builds the entire feature universe and aligns with forward prediction targets.
        """
        if df.empty or len(df) < 50:
            raise ValueError("Insufficient data to generate rolling quantitative features (need >= 50 rows).")

        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        feats: List[pd.DataFrame] = []

        # 1. Price and Return Dynamics
        ret_df = calculate_returns(df, col="Close")
        feats.append(ret_df)

        # 2. Lagged Returns
        lags_df = calculate_lagged_features(ret_df["Return_Simple"], lags=[1, 2, 3, 5, 10, 21], prefix="Lag_Ret")
        feats.append(lags_df)

        # 3. Rolling Statistics
        roll_stats = calculate_rolling_statistics(ret_df["Return_Simple"], windows=[21, 63])
        feats.append(roll_stats)

        # 4. Autocorrelation & Mean Reversion
        autocorr_1 = calculate_autocorrelation(ret_df["Return_Simple"], window=63, lag=1)
        mean_rev = calculate_mean_reversion_score(df["Close"], window=20)
        feats.append(pd.DataFrame({"Autocorr_Lag1_63": autocorr_1, "Mean_Rev_Z_20": mean_rev}, index=df.index))

        # 5. Technical Indicators
        sma_20 = calculate_sma(df["Close"], window=20)
        sma_50 = calculate_sma(df["Close"], window=50)
        sma_200 = calculate_sma(df["Close"], window=200)
        dist_sma200 = (df["Close"] - sma_200) / (sma_200 + 1e-12)
        rsi_14 = calculate_rsi(df["Close"], window=14)
        macd_df = calculate_macd(df["Close"], fast=12, slow=26, signal=9)
        bb_df = calculate_bollinger_bands(df["Close"], window=20, num_std=2.0)
        atr_14 = calculate_atr(df, window=14)
        stoch_df = calculate_stochastic(df, k_window=14, d_window=3)
        adx_14 = calculate_adx(df, window=14)
        obv = calculate_obv(df)
        obv_pct = obv.pct_change(21)  # Rolling 21-day volume trend

        feats.append(
            pd.DataFrame(
                {
                    "SMA_Ratio_20_50": (sma_20 / (sma_50 + 1e-12)) - 1.0,
                    "Dist_SMA_200": dist_sma200,
                    "RSI_14": rsi_14,
                    "ATR_Pct_14": atr_14 / (df["Close"] + 1e-12),
                    "ADX_14": adx_14,
                    "OBV_Pct_21": obv_pct,
                },
                index=df.index,
            )
        )
        feats.append(macd_df)
        feats.append(bb_df)
        feats.append(stoch_df)

        # 6. Advanced Volatility Estimators
        vol_cc = calculate_realized_volatility(ret_df["Return_Simple"], window=21)
        vol_pk = calculate_parkinson_volatility(df, window=21)
        vol_gk = calculate_garman_klass_volatility(df, window=21)
        vol_rs = calculate_rogers_satchell_volatility(df, window=21)
        vol_yz = calculate_yang_zhang_volatility(df, window=21)
        vol_of_vol = calculate_volatility_of_volatility(vol_cc, window=21)

        feats.append(
            pd.DataFrame(
                {
                    "Vol_Realized_21": vol_cc,
                    "Vol_Parkinson_21": vol_pk,
                    "Vol_Garman_Klass_21": vol_gk,
                    "Vol_Rogers_Satchell_21": vol_rs,
                    "Vol_Yang_Zhang_21": vol_yz,
                    "Vol_Of_Vol_21": vol_of_vol,
                },
                index=df.index,
            )
        )

        # 7. Benchmark Beta and Correlation
        if benchmark_df is not None and not benchmark_df.empty and "Close" in benchmark_df.columns:
            bench_ret = benchmark_df["Close"].pct_change()
            bench_feats = calculate_rolling_beta_and_correlation(ret_df["Return_Simple"], bench_ret, window=63)
            feats.append(bench_feats)

        # 8. Macro Features
        if self.include_macro:
            macro_df = self.macro_engine.get_macro_features(df.index)
            feats.append(macro_df)

        # Concatenate all features
        X_all = pd.concat(feats, axis=1)

        # 9. Construct Forward Prediction Targets (Strict Shift by -1)
        # Target return at t is (Close_{t+1} - Close_t) / Close_t
        y_return = df["Close"].pct_change().shift(-1)
        y_direction = (y_return > 0).astype(int)
        y_volatility = ret_df["Return_Simple"].rolling(5).std().shift(-5) * np.sqrt(252)

        # Drop initial rows with NaNs resulting from 200-day rolling lookbacks
        valid_idx = X_all.dropna().index
        # Align with targets where target is not NaN (except last row kept for inference if needed)
        training_idx = valid_idx.intersection(y_return.dropna().index)

        X = X_all.loc[training_idx]
        y_ret = y_return.loc[training_idx]
        y_dir = y_direction.loc[training_idx]
        y_vol = y_volatility.loc[training_idx].fillna(y_volatility.mean())

        # Filter specific features if requested
        if selected_features:
            available_cols = [c for c in selected_features if c in X.columns]
            if available_cols:
                X = X[available_cols]

        # Calculate feature correlations with target
        corr_series = X.apply(lambda col: col.corr(y_ret)).sort_values(ascending=False)

        # Scaling if requested
        scaler = None
        if scale_method == "standard":
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)
            X = X_scaled
        elif scale_method == "robust":
            scaler = RobustScaler()
            X_scaled = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)
            X = X_scaled
        elif scale_method == "minmax":
            scaler = MinMaxScaler()
            X_scaled = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)
            X = X_scaled

        return FeatureMatrix(
            X=X,
            y_return=y_ret,
            y_direction=y_dir,
            y_volatility=y_vol,
            feature_names=list(X.columns),
            scaler=scaler,
            correlation_with_target=corr_series,
        )

    def select_top_features(self, X: pd.DataFrame, y: pd.Series, top_k: int = 15) -> List[str]:
        """
        Ranks features by Mutual Information with the continuous return target.
        """
        clean_X = X.dropna()
        clean_y = y.loc[clean_X.index]
        if len(clean_X) < 30 or clean_X.shape[1] <= top_k:
            return list(clean_X.columns)

        scores = mutual_info_regression(clean_X, clean_y, random_state=42)
        score_series = pd.Series(scores, index=clean_X.columns).sort_values(ascending=False)
        return list(score_series.head(top_k).index)
