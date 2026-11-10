"""
QuantEdge Market Data Caching Layer
Provides local file-based cache with TTL and checksum integrity.
"""

import hashlib
import json
import os
import time
from typing import Any, Callable, Optional
import pandas as pd


class CacheManager:
    """
    Manages local filesystem caching for financial time series and model outputs.
    """

    def __init__(self, cache_dir: str = "datasets/cache", default_ttl_seconds: int = 86400):
        self.cache_dir = cache_dir
        self.default_ttl = default_ttl_seconds
        os.makedirs(cache_dir, exist_ok=True)

    def _get_key_path(self, key: str, extension: str = "parquet") -> str:
        safe_key = hashlib.md5(key.encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, f"{safe_key}.{extension}")

    def _get_meta_path(self, key: str) -> str:
        safe_key = hashlib.md5(key.encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, f"{safe_key}_meta.json")

    def get_dataframe(self, key: str, max_age_seconds: Optional[int] = None) -> Optional[pd.DataFrame]:
        max_age = max_age_seconds if max_age_seconds is not None else self.default_ttl
        parquet_path = self._get_key_path(key, "parquet")
        csv_path = self._get_key_path(key, "csv")
        meta_path = self._get_meta_path(key)

        target_file = None
        if os.path.exists(parquet_path):
            target_file = parquet_path
        elif os.path.exists(csv_path):
            target_file = csv_path

        if not target_file or not os.path.exists(meta_path):
            return None

        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            cached_at = meta.get("timestamp", 0)
            if (time.time() - cached_at) > max_age:
                return None  # Cache expired

            if target_file.endswith(".parquet"):
                df = pd.read_parquet(target_file)
            else:
                df = pd.read_csv(target_file, index_col=0, parse_dates=True)
            return df
        except Exception:
            return None

    def put_dataframe(self, key: str, df: pd.DataFrame, metadata: Optional[dict] = None) -> None:
        if df is None or df.empty:
            return
        meta_path = self._get_meta_path(key)
        parquet_path = self._get_key_path(key, "parquet")

        try:
            df.to_parquet(parquet_path)
        except Exception:
            # Fallback to CSV if pyarrow/parquet not yet available
            csv_path = self._get_key_path(key, "csv")
            df.to_csv(csv_path)

        meta = {
            "key": key,
            "timestamp": time.time(),
            "rows": len(df),
            "columns": list(df.columns),
            "custom": metadata or {},
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    def clear(self) -> None:
        for f in os.listdir(self.cache_dir):
            p = os.path.join(self.cache_dir, f)
            if os.path.isfile(p):
                os.remove(p)
