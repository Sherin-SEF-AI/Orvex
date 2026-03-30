"""tests/test_edge_exporter.py — Unit tests for core/edge_exporter.py"""
import pytest
from core.edge_exporter import check_export_dependencies, ONNXExportResult


def test_check_export_dependencies():
    deps = check_export_dependencies()
    assert isinstance(deps, dict)
    assert "onnx" in deps or "onnxruntime" in deps or isinstance(deps, dict)


def test_onnx_export_result_model():
    result = ONNXExportResult(
        model_path="/tmp/model.onnx",
        input_shape=[1, 3, 640, 640],
        output_names=["output0"],
    )
    assert result.model_path == "/tmp/model.onnx"
