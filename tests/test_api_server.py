"""tests/test_api_server.py — Unit tests for core/api_server.py"""
import pytest
from core.api_server import InferenceRequest, InferenceResult


def test_inference_request_model():
    req = InferenceRequest(image_path="/tmp/test.jpg", conf_threshold=0.5)
    assert req.image_path == "/tmp/test.jpg"
    assert req.conf_threshold == 0.5


def test_inference_result_model():
    result = InferenceResult(detections=[], inference_time_ms=42.0)
    assert result.detections == []
    assert result.inference_time_ms == 42.0
