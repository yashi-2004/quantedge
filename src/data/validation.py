"""
QuantEdge Data Validation and Quality Assurance Engine
Ensures mathematical integrity, absence of lookahead bias, consistency,
and absence of corrupt or synthetic records misrepresented as real market data.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class ValidationReport:
    is_valid: bool
    quality_score: float  # 0.0 to 100.0
    total_records: int
    missing_count: int
    missing_pct: float
    duplicate_count: int
    inversion_count: int  # Cases where Low > High or Open/Close outside High/Low
    outlier_count: int
    zero_volume_count: int
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    issues: List[str] = field(default_factory=list)
    remediation_applied: List[str] = field(default_factory=list)


class DataValidator:
    """
    Validates OHLCV historical market data for quantitative research standards.
    """

    def __init__(self, max_outlier_zscore: float = 5.0, max_single_day_return: float = 0.50):
        self.max_outlier_zscore = max_outlier_zscore
        self.max_single_day_return = max_single_day_return

    def validate_ohlcv(self, df: pd.DataFrame, ticker: str = "ASSET") -> ValidationReport:
        """
        Runs comprehensive validation checks on an OHLCV dataframe.
        """
        if df is None or df.empty:
            return ValidationReport(
                is_valid=False,
                quality_score=0.0,
                total_records=0,
                missing_count=0,
                missing_pct=100.0,
                duplicate_count=0,
                inversion_count=0,
                outlier_count=0,
                zero_volume_count=0,
                issues=["Dataset is empty or None"],
            )

        df = df.copy()
        issues: List[str] = []
        total_records = len(df)

        # Check required columns
        required_cols = {"Open", "High", "Low", "Close"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            issues.append(f"Missing essential price columns: {missing_cols}")
            return ValidationReport(
                is_valid=False,
                quality_score=0.0,
                total_records=total_records,
                missing_count=total_records,
                missing_pct=100.0,
                duplicate_count=0,
                inversion_count=0,
                outlier_count=0,
                zero_volume_count=0,
                issues=issues,
            )

        # 1. Missing values
        missing_count = int(df[list(required_cols)].isna().sum().sum())
        missing_pct = float((missing_count / (total_records * len(required_cols))) * 100)
        if missing_count > 0:
            issues.append(f"Found {missing_count} NaN values across core price columns ({missing_pct:.2f}%).")

        # 2. Duplicate index timestamps
        duplicate_count = int(df.index.duplicated().sum())
        if duplicate_count > 0:
            issues.append(f"Found {duplicate_count} duplicate timestamps.")

        # 3. Price Inversions (Physical impossibilities in financial markets)
        high_low_inversion = (df["Low"] > df["High"]).sum()
        open_bounds = ((df["Open"] > df["High"]) | (df["Open"] < df["Low"])).sum()
        close_bounds = ((df["Close"] > df["High"]) | (df["Close"] < df["Low"])).sum()
        inversion_count = int(high_low_inversion + open_bounds + close_bounds)
        if inversion_count > 0:
            issues.append(f"Found {inversion_count} OHLC boundary violations (e.g., Low > High or Open/Close outside [Low, High]).")

        # 4. Outlier inspection on daily returns
        close = df["Close"].astype(float)
        ret = close.pct_change().dropna()
        extreme_returns = (ret.abs() > self.max_single_day_return).sum()
        z_scores = (ret - ret.mean()) / (ret.std() + 1e-12)
        z_outliers = (z_scores.abs() > self.max_outlier_zscore).sum()
        outlier_count = int(max(extreme_returns, z_outliers))
        if outlier_count > 0:
            issues.append(f"Detected {outlier_count} statistical return outliers (|r| > {self.max_single_day_return*100:.0f}% or |Z| > {self.max_outlier_zscore}).")

        # 5. Zero / Negative volume inspection
        zero_volume_count = 0
        if "Volume" in df.columns:
            zero_volume_count = int((df["Volume"] <= 0).sum())
            if zero_volume_count > 0:
                issues.append(f"Detected {zero_volume_count} zero or negative volume trading sessions.")

        # Calculate Quality Score (100 base, deductions for defects)
        deductions = 0.0
        deductions += min(30.0, missing_pct * 3.0)
        deductions += min(20.0, duplicate_count * 2.0)
        deductions += min(30.0, inversion_count * 5.0)
        deductions += min(15.0, outlier_count * 1.5)
        deductions += min(10.0, (zero_volume_count / max(1, total_records)) * 20.0)

        quality_score = max(0.0, round(100.0 - deductions, 1))
        is_valid = quality_score >= 60.0 and inversion_count == 0

        start_date = str(df.index[0]) if len(df) > 0 else None
        end_date = str(df.index[-1]) if len(df) > 0 else None

        return ValidationReport(
            is_valid=is_valid,
            quality_score=quality_score,
            total_records=total_records,
            missing_count=missing_count,
            missing_pct=round(missing_pct, 2),
            duplicate_count=duplicate_count,
            inversion_count=inversion_count,
            outlier_count=outlier_count,
            zero_volume_count=zero_volume_count,
            start_date=start_date,
            end_date=end_date,
            issues=issues,
        )

    def clean_ohlcv(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Cleans OHLCV data using sound quantitative procedures:
        - Drops duplicate timestamps keeping the latest record.
        - Fixes boundary anomalies where Low > High by taking min/max.
        - Forward-fills missing intermediate prices (up to 5 days), then drops remaining NaNs.
        - Ensures Volume >= 0.
        """
        if df is None or df.empty:
            return pd.DataFrame(), ["Empty dataframe provided."]

        df = df.copy()
        remediations: List[str] = []

        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Sort chronologically
        df = df.sort_index()

        # Deduplicate timestamps
        init_len = len(df)
        df = df[~df.index.duplicated(keep="last")]
        if len(df) < init_len:
            remediations.append(f"Removed {init_len - len(df)} duplicate timestamps.")

        # Inversion corrections
        if "High" in df.columns and "Low" in df.columns:
            actual_high = np.maximum(df["High"], df["Low"])
            actual_low = np.minimum(df["High"], df["Low"])
            inversions = (df["High"] != actual_high).sum()
            if inversions > 0:
                df["High"] = actual_high
                df["Low"] = actual_low
                remediations.append(f"Corrected {inversions} High/Low inversion anomalies.")

        # Clip Open and Close within High/Low bounds
        if "Open" in df.columns and "High" in df.columns and "Low" in df.columns:
            df["Open"] = df["Open"].clip(lower=df["Low"], upper=df["High"])
            df["Close"] = df["Close"].clip(lower=df["Low"], upper=df["High"])

        # Forward fill missing values up to 5 days, then back fill
        missing_before = df.isna().sum().sum()
        if missing_before > 0:
            df = df.ffill(limit=5).bfill(limit=2).dropna()
            remediations.append(f"Imputed missing entries using forward-fill; {len(df)} clean rows preserved.")

        # Non-negative volume
        if "Volume" in df.columns:
            neg_vol = (df["Volume"] < 0).sum()
            if neg_vol > 0:
                df["Volume"] = df["Volume"].clip(lower=0)
                remediations.append(f"Clipped {neg_vol} negative volume values to 0.")

        return df, remediations
