"""Tracking ligero con MLflow file-store (ADR §8, §19.15).

Fuente de verdad = ficheros JSON/parquet versionables. MLflow es un espejo opcional: si no está
instalado o falla, no rompe el experimento (no-op).
"""

from __future__ import annotations

import os


def log_run(experiment: str, run_name: str, params: dict, metrics: dict) -> None:
    try:
        import mlflow
    except ImportError:
        return
    try:
        # MLflow 3.x deprecó el file-store → backend sqlite por defecto (como el compose del ADR).
        uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///outputs/mlflow.db")
        if uri.startswith("sqlite:///"):
            os.makedirs(os.path.dirname(uri.replace("sqlite:///", "")) or ".", exist_ok=True)
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({k: str(v) for k, v in params.items()})
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
    except Exception:
        # MLflow es un espejo; cualquier fallo de tracking no debe abortar el experimento.
        return
