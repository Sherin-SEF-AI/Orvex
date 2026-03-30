"""
web/backend/routes/edge_export.py — Edge model export endpoints.
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

router = APIRouter(prefix="/edge-export", tags=["edge-export"])


class ONNXRequest(BaseModel):
    weights_path: str
    output_path: str
    image_size: int = 640
    batch_size: int = 1
    simplify: bool = True
    opset_version: int = 17


class TRTRequest(BaseModel):
    onnx_path: str
    output_path: str
    precision: str = "fp16"
    workspace_gb: int = 4


class BenchmarkRequest(BaseModel):
    model_path: str
    model_format: str = "onnx"
    image_size: int = 640
    n_iterations: int = 200


class PackageRequest(BaseModel):
    weights_path: str
    onnx_path: str
    trt_path: str | None = None
    class_names: list[str]
    output_dir: str
    target_device: str = "jetson_orin"
    conf_threshold: float = 0.25


@router.get("/dependencies", summary="Check export dependencies")
def check_deps() -> dict:
    from core.edge_exporter import check_export_dependencies
    return {"data": check_export_dependencies(), "error": None}


@router.post("/onnx", summary="Export model to ONNX")
def export_onnx(body: ONNXRequest) -> dict:
    try:
        from core.edge_exporter import export_to_onnx
        result = export_to_onnx(
            body.weights_path, body.output_path,
            body.image_size, body.batch_size,
            body.simplify, body.opset_version,
        )
        return {"data": result.model_dump(), "error": None}
    except (ImportError, RuntimeError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/tensorrt", summary="Convert ONNX to TensorRT engine")
def export_trt(body: TRTRequest) -> dict:
    try:
        from core.edge_exporter import export_to_tensorrt
        result = export_to_tensorrt(
            body.onnx_path, body.output_path,
            body.precision, body.workspace_gb,
        )
        return {"data": result.model_dump(), "error": None}
    except (RuntimeError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/benchmark", summary="Benchmark model inference")
def benchmark(body: BenchmarkRequest) -> dict:
    try:
        from core.edge_exporter import benchmark_model
        result = benchmark_model(
            body.model_path, body.model_format,
            body.image_size, body.n_iterations,
        )
        return {"data": result.model_dump(), "error": None}
    except (ImportError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/package", summary="Build Jetson deployment package")
def package(body: PackageRequest) -> dict:
    try:
        from core.edge_exporter import package_jetson_deployment
        tar_path = package_jetson_deployment(
            body.weights_path, body.onnx_path, body.trt_path,
            body.class_names, body.output_dir,
            body.target_device, body.conf_threshold,
        )
        return {"data": {"tar_path": tar_path}, "error": None}
    except (ImportError, RuntimeError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/package/download", summary="Download deployment package .tar.gz")
def download_package(tar_path: str) -> FileResponse:
    from pathlib import Path
    p = Path(tar_path)
    if not p.exists():
        raise HTTPException(status_code=404, detail="Package not found")
    return FileResponse(str(p), filename=p.name, media_type="application/gzip")
