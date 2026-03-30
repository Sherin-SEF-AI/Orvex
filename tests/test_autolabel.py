"""tests/test_autolabel.py — Unit tests for core/autolabel.py"""
import pytest
from core.autolabel import FrameAnnotation, Detection


def test_detection_model():
    det = Detection(label="car", confidence=0.95, bbox=[0.1, 0.2, 0.3, 0.4])
    assert det.label == "car"
    assert det.confidence == 0.95


def test_frame_annotation_model():
    ann = FrameAnnotation(frame_path="/tmp/frame.jpg", detections=[])
    assert ann.frame_path == "/tmp/frame.jpg"
    assert ann.detections == []
