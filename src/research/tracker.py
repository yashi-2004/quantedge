"""
QuantEdge Local Experiment Tracking Engine
Wraps local MLflow with SQLite backend to log runs, parameters, metrics, and model artifacts.
"""

from dataclasses import asdict, dataclass
import json
import os
import time
from typing import Any, Dict, List, Optional
import mlflow
import pandas as pd


class ExperimentTracker:
    """
    Manages local experiment logging without external cloud dependencies.
    """

    def __init__(self, db_path: str = "experiments/mlflow.db", default_experiment: str = "QuantEdge_Research"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.tracking_uri = f"sqlite:///{os.path.abspath(db_path)}"
        mlflow.set_tracking_uri(self.tracking_uri)
        self.experiment_name = default_experiment
        mlflow.set_experiment(default_experiment)

    def log_run(
        self,
        run_name: str,
        model_type: str,
        parameters: Dict[str, Any],
        metrics: Dict[str, float],
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Logs a single quantitative modeling or backtesting experiment run.
        """
        all_tags = {"model_type": model_type, **(tags or {})}
        with mlflow.start_run(run_name=run_name) as run:
            for k, v in parameters.items():
                mlflow.log_param(k, str(v)[:250])
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and not np.isnan(v):
                    mlflow.log_metric(k, float(v))
            mlflow.set_tags(all_tags)
            return run.info.run_id

    def list_experiments(self) -> List[str]:
        return [exp.name for exp in mlflow.search_experiments()]

    def get_runs_dataframe(self, experiment_name: Optional[str] = None) -> pd.DataFrame:
        exp_name = experiment_name or self.experiment_name
        exp = mlflow.get_experiment_by_name(exp_name)
        if exp is None:
            return pd.DataFrame()
        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
        return runs
