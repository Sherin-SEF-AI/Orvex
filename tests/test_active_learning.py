"""tests/test_active_learning.py — Unit tests for core/active_learning.py"""
import pytest
from core.active_learning import compute_uncertainty_scores, UncertaintyScore


def test_uncertainty_score_model():
    score = UncertaintyScore(image_id="img1", score=0.85, strategy="entropy")
    assert score.image_id == "img1"
    assert score.score == 0.85


def test_compute_uncertainty_scores_empty():
    result = compute_uncertainty_scores([], strategy="entropy")
    assert result == []
