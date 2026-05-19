"""MLflow housekeeping: retention, disk checks, registry pruning."""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Optional

import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parent
MLRUNS_DIR = BACKEND_DIR / "mlruns"
MIN_FREE_BYTES = int(os.getenv("MLFLOW_MIN_FREE_MB", "500")) * 1024 * 1024


def disk_free_bytes(path: Path | str = BACKEND_DIR) -> int:
    usage = shutil.disk_usage(Path(path).anchor if os.name == "nt" else path)
    return usage.free


def mlruns_size_bytes() -> int:
    if not MLRUNS_DIR.exists():
        return 0
    return sum(f.stat().st_size for f in MLRUNS_DIR.rglob("*") if f.is_file())


def ensure_disk_space(path: Path | str = BACKEND_DIR) -> None:
    free = disk_free_bytes(path)
    if free < MIN_FREE_BYTES:
        raise OSError(
            f"Espace disque insuffisant ({free // (1024 * 1024)} Mo libres, "
            f"minimum {MIN_FREE_BYTES // (1024 * 1024)} Mo). "
            "Appelez POST /maintenance/cleanup ou supprimez automl-backend/mlruns."
        )


def prune_experiment_runs(
    client: MlflowClient,
    experiment_id: str,
    keep_last: int = 20,
) -> int:
    runs_df = mlflow.search_runs(
        experiment_ids=[experiment_id],
        order_by=["start_time DESC"],
        max_results=10_000,
    )
    if runs_df.empty or len(runs_df) <= keep_last:
        return 0
    deleted = 0
    for run_id in runs_df.iloc[keep_last:]["run_id"]:
        try:
            client.delete_run(run_id)
            deleted += 1
        except Exception as exc:
            logger.warning("Could not delete run %s: %s", run_id, exc)
    return deleted


def prune_registered_model_versions(
    client: MlflowClient,
    keep_versions: int = 3,
) -> int:
    deleted = 0
    try:
        models = client.search_registered_models()
    except Exception:
        return 0
    for model in models:
        try:
            versions = client.search_model_versions(f"name='{model.name}'")
        except Exception:
            continue
        sorted_v = sorted(versions, key=lambda v: int(v.version), reverse=True)
        for ver in sorted_v[keep_versions:]:
            try:
                client.delete_model_version(model.name, ver.version)
                deleted += 1
            except Exception as exc:
                logger.warning("Could not delete %s v%s: %s", model.name, ver.version, exc)
    return deleted


def cleanup_mlflow_store(
    client: Optional[MlflowClient] = None,
    keep_runs_per_experiment: int = 20,
    keep_model_versions: int = 3,
) -> dict:
    """Prune old runs and registry versions to reclaim disk space."""
    client = client or MlflowClient()
    runs_deleted = 0
    versions_deleted = 0

    for exp in mlflow.search_experiments():
        runs_deleted += prune_experiment_runs(
            client, exp.experiment_id, keep_last=keep_runs_per_experiment
        )

    versions_deleted = prune_registered_model_versions(client, keep_versions=keep_model_versions)

    return {
        "runs_deleted": runs_deleted,
        "model_versions_deleted": versions_deleted,
        "mlruns_size_mb": round(mlruns_size_bytes() / (1024 * 1024), 2),
        "disk_free_mb": round(disk_free_bytes() / (1024 * 1024), 2),
    }


def reset_mlflow_artifacts() -> dict:
    """Delete local mlruns and mlflow.db (destructive). Use when disk is full."""
    removed = []
    db_path = BACKEND_DIR / "mlflow.db"
    if MLRUNS_DIR.exists():
        shutil.rmtree(MLRUNS_DIR, ignore_errors=True)
        removed.append(str(MLRUNS_DIR))
    for suffix in ("", "-wal", "-shm", "-journal"):
        p = Path(f"{db_path}{suffix}")
        if p.exists():
            p.unlink(missing_ok=True)
            removed.append(str(p))
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    return {"reset": True, "removed": removed}
