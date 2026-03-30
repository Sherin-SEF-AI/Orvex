"""tests/test_experiment_tracker.py — Unit tests for core/experiment_tracker.py"""
import pytest
from core.experiment_tracker import check_mlflow_installation, MLflowRun


def test_check_mlflow_installation():
    result = check_mlflow_installation()
    assert isinstance(result, bool)


def test_mlflow_run_model():
    run = MLflowRun(
        run_id="abc123",
        experiment_name="test",
        status="FINISHED",
        metrics={"mAP": 0.85},
        params={"lr": "0.001"},
    )
    assert run.run_id == "abc123"
    assert run.metrics["mAP"] == 0.85
