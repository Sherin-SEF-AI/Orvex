"""tests/test_trainer.py — Unit tests for core/trainer.py"""
import pytest
from core.trainer import TrainingConfig, TrainingRun, EpochMetrics


def test_training_config_defaults():
    cfg = TrainingConfig(model="yolov8n.pt", data="/tmp/data.yaml")
    assert cfg.model == "yolov8n.pt"


def test_epoch_metrics_model():
    metrics = EpochMetrics(
        epoch=1,
        train_loss=2.5,
        val_loss=2.8,
        mAP50=0.45,
    )
    assert metrics.epoch == 1
    assert metrics.mAP50 == 0.45


def test_training_run_model():
    run = TrainingRun(
        run_id="run_001",
        config=TrainingConfig(model="yolov8n.pt", data="/tmp/data.yaml"),
        status="pending",
    )
    assert run.run_id == "run_001"
    assert run.status == "pending"
