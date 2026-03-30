"""
tests/test_inference_server.py — Unit tests for core/inference_server.py.

Tests that require trained weights are skipped if no .pt file is present
in tests/fixtures/.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from core.inference_server import (
    delete_model,
    get_active_model,
    load_registry,
    register_model,
    save_registry,
    set_active_model,
)
from core.models import ModelRegistry

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_registry(tmp_path):
    """Return a temporary registry TOML path."""
    return tmp_path / "registry.toml"


@pytest.fixture
def dummy_weights(tmp_path):
    """Create a zero-byte .pt file to satisfy FileNotFoundError checks."""
    p = tmp_path / "dummy.pt"
    p.write_bytes(b"")
    return p


# ---------------------------------------------------------------------------
# Registry CRUD
# ---------------------------------------------------------------------------

def test_load_empty_registry(tmp_registry):
    assert load_registry(tmp_registry) == []


def test_register_and_load(tmp_registry, dummy_weights):
    entry = register_model(
        weights_path=str(dummy_weights),
        name="test_model",
        model_variant="yolov8n",
        registry_path=tmp_registry,
    )
    assert entry.name == "test_model"
    assert entry.model_variant == "yolov8n"
    assert entry.is_active is True   # first model auto-activated

    models = load_registry(tmp_registry)
    assert len(models) == 1
    assert models[0].model_id == entry.model_id


def test_register_second_not_active(tmp_registry, dummy_weights):
    register_model("", "first", "yolov8n", registry_path=tmp_registry)
    # Create second weights file
    w2 = dummy_weights.parent / "second.pt"
    w2.write_bytes(b"")

    # Patch register_model to avoid FileNotFoundError for the first (empty string)
    # Instead register both with real files
    tmp_registry.unlink(missing_ok=True)
    e1 = register_model(str(dummy_weights), "first",  "yolov8n", registry_path=tmp_registry)
    e2 = register_model(str(w2),            "second", "yolov8s", registry_path=tmp_registry)
    assert e1.is_active is True
    assert e2.is_active is False


def test_set_active_model(tmp_registry, dummy_weights):
    w2 = dummy_weights.parent / "second.pt"
    w2.write_bytes(b"")
    e1 = register_model(str(dummy_weights), "first",  "yolov8n", registry_path=tmp_registry)
    e2 = register_model(str(w2),            "second", "yolov8s", registry_path=tmp_registry)

    set_active_model(e2.model_id, tmp_registry)
    models = load_registry(tmp_registry)
    active = [m for m in models if m.is_active]
    assert len(active) == 1
    assert active[0].model_id == e2.model_id


def test_get_active_model(tmp_registry, dummy_weights):
    entry = register_model(str(dummy_weights), "only", "yolov8n", registry_path=tmp_registry)
    active = get_active_model(tmp_registry)
    assert active.model_id == entry.model_id


def test_get_active_model_raises_when_none(tmp_registry):
    with pytest.raises(RuntimeError, match="No active model"):
        get_active_model(tmp_registry)


def test_delete_model(tmp_registry, dummy_weights):
    entry = register_model(str(dummy_weights), "to_delete", "yolov8n", registry_path=tmp_registry)
    delete_model(entry.model_id, tmp_registry)
    assert load_registry(tmp_registry) == []


def test_delete_unknown_raises(tmp_registry):
    with pytest.raises(ValueError, match="not found"):
        delete_model("nonexistent-id", tmp_registry)


def test_register_missing_weights_raises(tmp_registry):
    with pytest.raises(FileNotFoundError):
        register_model("/no/such/file.pt", "bad", "yolov8n", registry_path=tmp_registry)


# ---------------------------------------------------------------------------
# Registry TOML round-trip
# ---------------------------------------------------------------------------

def test_registry_toml_roundtrip(tmp_registry, dummy_weights):
    entry = register_model(
        str(dummy_weights), "roundtrip", "yolov8n",
        metrics={"map50": 0.75, "map50_95": 0.45},
        registry_path=tmp_registry,
    )
    reloaded = load_registry(tmp_registry)
    assert reloaded[0].metrics["map50"] == pytest.approx(0.75)
    assert reloaded[0].is_active is True


# ---------------------------------------------------------------------------
# Skip-if-no-weights inference test
# ---------------------------------------------------------------------------

FIXTURE_WEIGHTS = Path(__file__).parent / "fixtures" / "test_model.pt"

@pytest.mark.skipif(
    not FIXTURE_WEIGHTS.exists(),
    reason="tests/fixtures/test_model.pt not present — skipping live inference test",
)
def test_run_inference(tmp_registry, tmp_path):
    """Integration test: load real weights and run inference on a black image."""
    import numpy as np
    from PIL import Image
    from core.inference_server import run_inference
    from core.models import InferenceRequest

    # Register fixture weights
    register_model(str(FIXTURE_WEIGHTS), "fixture", "yolov8n", registry_path=tmp_registry)

    # Create a tiny black JPEG
    img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    img_path = tmp_path / "test.jpg"
    img.save(str(img_path))

    result = run_inference(
        InferenceRequest(image_path=str(img_path), conf_threshold=0.01),
        registry_path=tmp_registry,
    )
    assert result.inference_time_ms > 0
    assert result.model_variant == "yolov8n"
