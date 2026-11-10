"""
QuantEdge Local Persistence & Storage Engine
Uses SQLite for robust zero-cost local metadata, caching, and experiment data.
"""

import os
import sqlite3
from typing import Dict, List, Optional
import pandas as pd


class LocalStorage:
    """
    Manages SQLite database storage for market data, metadata, and research artifacts.
    """

    def __init__(self, db_path: str = "datasets/cache/quantedge_data.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_data_registry (
                    ticker TEXT PRIMARY KEY,
                    asset_type TEXT,
                    currency TEXT,
                    start_date TEXT,
                    end_date TEXT,
                    row_count INTEGER,
                    quality_score REAL,
                    provenance TEXT,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiment_runs (
                    run_id TEXT PRIMARY KEY,
                    experiment_name TEXT,
                    model_type TEXT,
                    target TEXT,
                    metrics_json TEXT,
                    parameters_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def save_market_data(
        self,
        ticker: str,
        df: pd.DataFrame,
        provenance: str = "LIVE_YFINANCE",
        quality_score: float = 100.0,
        asset_type: str = "EQUITY",
        currency: str = "USD",
    ) -> None:
        """
        Saves clean OHLCV data to SQLite table named `data_{sanitized_ticker}`
        and updates the registry.
        """
        if df.empty:
            return

        clean_table_name = f"data_{ticker.replace('.', '_').replace('-', '_').upper()}"
        df_to_save = df.copy()
        if not isinstance(df_to_save.index, pd.DatetimeIndex):
            df_to_save.index = pd.to_datetime(df_to_save.index)
        df_to_save["Date"] = df_to_save.index.strftime("%Y-%m-%d")

        with self._get_connection() as conn:
            df_to_save.to_sql(clean_table_name, conn, if_exists="replace", index=False)
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO market_data_registry 
                (ticker, asset_type, currency, start_date, end_date, row_count, quality_score, provenance, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
                (
                    ticker.upper(),
                    asset_type,
                    currency,
                    str(df_to_save["Date"].min()),
                    str(df_to_save["Date"].max()),
                    len(df_to_save),
                    float(quality_score),
                    provenance,
                ),
            )
            conn.commit()

    def load_market_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Loads OHLCV data from SQLite table if present.
        """
        clean_table_name = f"data_{ticker.replace('.', '_').replace('-', '_').upper()}"
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (clean_table_name,),
            )
            if not cursor.fetchone():
                return None

            df = pd.read_sql_query(f"SELECT * FROM {clean_table_name}", conn)
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])
                df.set_index("Date", inplace=True)
            return df

    def get_registered_assets(self) -> pd.DataFrame:
        with self._get_connection() as conn:
            return pd.read_sql_query("SELECT * FROM market_data_registry", conn)
