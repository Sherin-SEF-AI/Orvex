"""
core/experiment_tracker.py — MLflow experiment tracking for RoverDataKit.

MLflow is optional.  All mlflow imports are wrapped in try/except ImportError.
If mlflow is not installed every function logs a warning and returns gracefully.

Install with:  pip install mlflow
"""
from __future__ import annotations

import os
import subprocess
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from core.models import EpochMetrics, MLflowRun, RunComparison, TrainingConfig, TrainingRun

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_MLFLOW_MISSING_MSG = (
    "mlflow not installed. Install with: pip install mlflow"
)


def _mlflow():
    """Return the mlflow module or raise ImportError with an actionable message."""
    try:
        import mlflow
        return mlflow
    except ImportError as exc:
        raise ImportError(_MLFLOW_MISSING_MSG) from exc


def _warn_no_mlflow() -> None:
    logger.warning(_MLFLOW_MISSING_MSG)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_mlflow_installation() -> bool:
    """Return True if mlflow is importable."""
    try:
        import mlflow  # noqa: F401
        return True
    except ImportError:
        return False


def init_mlflow(
    tracking_uri: str = "sqlite:///data/mlflow.db",
    experiment_name: str = "rover_detection",
) -> str | None:
    """
    Initialize MLflow with a SQLite backend.

    Converts a relative ``sqlite:///`` URI to an absolute path so the database
    is always placed at the same location regardless of the working directory.
    Creates the experiment if it does not already exist.

    Returns:
        The experiment_id string, or None if mlflow is not installed.
    """
    if not check_mlflow_installation():
        _warn_no_mlflow()
        return None

    mlflow = _mlflow()

    # Resolve relative sqlite:/// path to absolute
    if tracking_uri.startswith("sqlite:///") and not tracking_uri.startswith("sqlite:////"):
        rel_path = tracking_uri[len("sqlite:///"):]
        abs_path = str(Path(rel_path).resolve())
        # Ensure parent directory exists
        Path(abs_path).parent.mkdir(parents=True, exist_ok=True)
        tracking_uri = f"sqlite:///{abs_path}"

    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"MLflow tracking URI: {tracking_uri}")

    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            logger.info(f"Created MLflow experiment '{experiment_name}' (id={experiment_id})")
        else:
            experiment_id = experiment.experiment_id
            logger.info(
                f"Using existing MLflow experiment '{experiment_name}' (id={experiment_id})"
            )
        return experiment_id
    except Exception as exc:
        logger.error(f"Failed to initialise MLflow experiment: {exc}")
        return None


def start_training_run(
    experiment_name: str,
    run_name: str,
    config: TrainingConfig,
    dataset_version: str | None = None,
) -> str | None:
    """
    Start a new MLflow run under *experiment_name*.

    Logs all ``TrainingConfig`` fields as params.  If *dataset_version* is
    supplied it is attached as the tag ``dataset_version``.

    Returns:
        The run_id string, or None if mlflow is not available.
    """
    if not check_mlflow_installation():
        _warn_no_mlflow()
        return None

    mlflow = _mlflow()

    try:
        mlflow.set_experiment(experiment_name)
        run = mlflow.start_run(run_name=run_name)
        run_id = run.info.run_id

        # Log every TrainingConfig field as a param
        params = config.model_dump()
        mlflow.log_params(params)

        if dataset_version is not None:
            mlflow.set_tag("dataset_version", dataset_version)

        mlflow.set_tag("run_name", run_name)
        logger.info(f"Started MLflow run '{run_name}' (id={run_id})")
        return run_id

    except Exception as exc:
        logger.error(f"Failed to start MLflow run: {exc}")
        return None


def log_epoch_metrics(run_id: str, metrics: EpochMetrics) -> None:
    """
    Log per-epoch training metrics to an active MLflow run.

    Metric keys follow the namespaced convention used by training dashboards:
    ``train/box_loss``, ``train/cls_loss``, ``train/dfl_loss``,
    ``val/precision``, ``val/recall``, ``val/mAP50``, ``val/mAP50-95``, ``lr``.

    Args:
        run_id: The MLflow run_id returned by :func:`start_training_run`.
        metrics: Populated ``EpochMetrics`` instance for this epoch.
    """
    if not check_mlflow_installation():
        _warn_no_mlflow()
        return

    mlflow = _mlflow()

    try:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics(
                {
                    "train/box_loss": metrics.box_loss,
                    "train/cls_loss": metrics.cls_loss,
                    "train/dfl_loss": metrics.dfl_loss,
                    "val/precision": metrics.precision,
                    "val/recall": metrics.recall,
                    "val/mAP50": metrics.map50,
                    "val/mAP50-95": metrics.map50_95,
                    "lr": metrics.lr,
                },
                step=metrics.epoch,
            )
    except Exception as exc:
        logger.warning(f"Failed to log epoch metrics to MLflow (run={run_id}): {exc}")


def finish_training_run(
    run_id: str,
    result: TrainingRun,
    weights_path: str,
    confusion_matrix_path: str | None = None,
) -> None:
    """
    Finalise an MLflow run after training completes.

    Logs best validation metrics, uploads the weights file as an artifact, and
    optionally uploads the confusion matrix image.  Sets the run status to
    ``FINISHED`` or ``FAILED`` based on ``result.status``.

    Args:
        run_id: The MLflow run_id to finish.
        result: Completed ``TrainingRun`` instance.
        weights_path: Absolute path to the best weights file (``best.pt``).
        confusion_matrix_path: Optional path to a confusion-matrix image.
    """
    if not check_mlflow_installation():
        _warn_no_mlflow()
        return

    mlflow = _mlflow()

    try:
        from mlflow.entities import RunStatus  # type: ignore[import-untyped]
        with mlflow.start_run(run_id=run_id):
            # Best final metrics
            mlflow.log_metrics(
                {
                    "best/mAP50": result.final_map50,
                    "best/mAP50-95": result.final_map50_95,
                    "best/epoch": float(result.best_epoch),
                    "training_time_minutes": result.training_time_minutes,
                }
            )

            # Artifacts
            weights_file = Path(weights_path)
            if weights_file.exists():
                mlflow.log_artifact(str(weights_file), artifact_path="weights")
                logger.info(f"Logged weights artifact: {weights_file.name}")
            else:
                logger.warning(f"Weights file not found, skipping artifact log: {weights_path}")

            if confusion_matrix_path is not None:
                cm_file = Path(confusion_matrix_path)
                if cm_file.exists():
                    mlflow.log_artifact(str(cm_file), artifact_path="plots")
                else:
                    logger.warning(
                        f"Confusion matrix file not found, skipping: {confusion_matrix_path}"
                    )

            # Terminate run with appropriate status
            terminal_status = (
                RunStatus.to_string(RunStatus.FINISHED)
                if result.status == "done"
                else RunStatus.to_string(RunStatus.FAILED)
            )
            mlflow.end_run(status=terminal_status)
            logger.info(f"Finished MLflow run {run_id} with status={terminal_status}")

    except Exception as exc:
        logger.error(f"Failed to finish MLflow run {run_id}: {exc}")
        try:
            mlflow.end_run(status="FAILED")
        except Exception:
            pass


def get_all_runs(
    experiment_name: str = "rover_detection",
    metric_sort: str = "val/mAP50",
) -> list[MLflowRun]:
    """
    Fetch all runs for *experiment_name* sorted by *metric_sort* descending.

    Returns:
        List of ``MLflowRun`` instances (may be empty).  Returns ``[]`` if
        mlflow is not installed or the experiment does not exist.
    """
    if not check_mlflow_installation():
        _warn_no_mlflow()
        return []

    mlflow = _mlflow()

    try:
        from datetime import datetime, timezone

        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            logger.info(f"MLflow experiment '{experiment_name}' does not exist yet.")
            return []

        runs_df = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.`{metric_sort}` DESC"],
        )

        result: list[MLflowRun] = []
        for _, row in runs_df.iterrows():
            # Extract params, metrics, and tags from flattened column names
            params: dict = {}
            metrics: dict = {}
            tags: dict = {}

            for col in runs_df.columns:
                val = row[col]
                if col.startswith("params."):
                    params[col[len("params."):]] = val
                elif col.startswith("metrics."):
                    metrics[col[len("metrics."):]] = val
                elif col.startswith("tags."):
                    tags[col[len("tags."):]] = val

            # start_time is an integer (epoch ms) in the dataframe
            raw_start = row.get("start_time", 0)
            if raw_start and raw_start == raw_start:  # not NaN
                start_dt = datetime.fromtimestamp(int(raw_start) / 1000, tz=timezone.utc)
            else:
                start_dt = datetime.now(tz=timezone.utc)

            run_name = tags.get("mlflow.runName", row.get("run_id", ""))

            result.append(
                MLflowRun(
                    run_id=row.get("run_id", ""),
                    run_name=run_name,
                    status=row.get("status", "UNKNOWN"),
                    start_time=start_dt,
                    params=params,
                    metrics=metrics,
                    tags=tags,
                    artifact_uri=row.get("artifact_uri", ""),
                )
            )

        logger.info(f"Fetched {len(result)} MLflow runs from '{experiment_name}'")
        return result

    except Exception as exc:
        logger.error(f"Failed to fetch MLflow runs: {exc}")
        return []


def compare_runs(run_ids: list[str]) -> RunComparison | None:
    """
    Compare a set of MLflow runs.

    Determines the best run by ``val/mAP50`` and computes which params differ
    between *any* two runs in the set.

    Returns:
        A ``RunComparison`` instance, or None if mlflow is not available or
        fewer than 2 run_ids are provided.
    """
    if not check_mlflow_installation():
        _warn_no_mlflow()
        return None

    if len(run_ids) < 2:
        logger.warning("compare_runs requires at least 2 run_ids.")
        return None

    mlflow = _mlflow()

    try:
        runs: list[MLflowRun] = []
        for rid in run_ids:
            run_data = mlflow.get_run(rid)
            if run_data is None:
                logger.warning(f"Run {rid} not found in MLflow — skipping.")
                continue

            from datetime import datetime, timezone

            info = run_data.info
            data = run_data.data
            start_dt = datetime.fromtimestamp(
                int(info.start_time) / 1000, tz=timezone.utc
            )

            runs.append(
                MLflowRun(
                    run_id=info.run_id,
                    run_name=data.tags.get("mlflow.runName", info.run_id),
                    status=info.status,
                    start_time=start_dt,
                    params=dict(data.params),
                    metrics=dict(data.metrics),
                    tags=dict(data.tags),
                    artifact_uri=info.artifact_uri,
                )
            )

        if len(runs) < 2:
            logger.warning("Fewer than 2 valid runs found; cannot compare.")
            return None

        # Determine best run by val/mAP50
        best_run = max(runs, key=lambda r: float(r.metrics.get("val/mAP50", 0.0)))

        # Build metric_comparison: each metric → {run_id: value}
        all_metric_keys: set[str] = set()
        for r in runs:
            all_metric_keys.update(r.metrics.keys())

        metric_comparison: dict = {
            key: {r.run_id: r.metrics.get(key) for r in runs}
            for key in sorted(all_metric_keys)
        }

        # Build param_differences: params that differ across any two runs
        all_param_keys: set[str] = set()
        for r in runs:
            all_param_keys.update(r.params.keys())

        param_differences: dict = {}
        for key in sorted(all_param_keys):
            values = {r.run_id: r.params.get(key) for r in runs}
            unique_vals = set(v for v in values.values() if v is not None)
            if len(unique_vals) > 1:
                param_differences[key] = values

        logger.info(
            f"Compared {len(runs)} runs; best={best_run.run_id} "
            f"(mAP50={best_run.metrics.get('val/mAP50', '—')})"
        )

        return RunComparison(
            runs=runs,
            best_run_id=best_run.run_id,
            metric_comparison=metric_comparison,
            param_differences=param_differences,
        )

    except Exception as exc:
        logger.error(f"Failed to compare MLflow runs: {exc}")
        return None


def launch_mlflow_ui(port: int = 5000) -> "subprocess.Popen | None":
    """
    Launch the MLflow tracking UI in a background subprocess.

    Opens the browser at ``http://localhost:{port}`` after a 2-second delay
    (using ``threading.Timer`` so the call returns immediately).

    Returns:
        The ``subprocess.Popen`` handle, or None on failure.
    """
    import webbrowser

    db_path = str(Path("data/mlflow.db").resolve())
    cmd = [
        "mlflow",
        "ui",
        "--backend-store-uri",
        f"sqlite:///{db_path}",
        "--port",
        str(port),
    ]
    logger.info(f"Launching MLflow UI: {' '.join(cmd)}")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        logger.error(
            "mlflow executable not found. "
            "Install with: pip install mlflow  and ensure the virtualenv is active."
        )
        return None
    except Exception as exc:
        logger.error(f"Failed to launch MLflow UI: {exc}")
        return None

    url = f"http://localhost:{port}"

    def _open_browser() -> None:
        logger.info(f"Opening MLflow UI at {url}")
        webbrowser.open(url)

    threading.Timer(2.0, _open_browser).start()
    return proc
