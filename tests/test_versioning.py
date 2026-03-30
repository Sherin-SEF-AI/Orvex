"""tests/test_versioning.py — Unit tests for core/versioning.py"""
import pytest
from core.versioning import check_dvc_installation, DatasetVersion


def test_check_dvc_installation():
    result = check_dvc_installation()
    assert isinstance(result, bool)


def test_dataset_version_model():
    version = DatasetVersion(
        tag="v1.0",
        message="Initial dataset",
        timestamp="2024-01-01T00:00:00Z",
    )
    assert version.tag == "v1.0"
    assert version.message == "Initial dataset"
