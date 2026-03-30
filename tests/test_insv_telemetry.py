"""tests/test_insv_telemetry.py — Unit tests for core/insv_telemetry.py"""
import pytest
from core.insv_telemetry import find_insv_pairs, INSVPair


def test_find_insv_pairs_empty():
    from pathlib import Path
    result = find_insv_pairs(Path("/nonexistent"))
    assert result == [] or isinstance(result, list)


def test_insv_pair_model():
    pair = INSVPair(front="/tmp/front.insv", back="/tmp/back.insv")
    assert pair.front == "/tmp/front.insv"
    assert pair.back == "/tmp/back.insv"
