"""
labelox/core/exporter.py — Export annotations in YOLO, COCO, CVAT XML, VOC, CSV formats.
"""
from __future__ import annotations

import json
import random
import shutil
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Callable
from xml.dom import minidom

from loguru import logger
from sqlalchemy.orm import Session

from labelox.core.annotation_engine import bbox_to_abs, bbox_to_yolo, get_annotations
from labelox.core.database import DBImage, DBProject, get_images, get_session
from labelox.core.models import (
    Annotation,
    AnnotationType,
    ExportConfig,
    ExportFormat,
    ExportResult,
    ImageStatus,
)

ProgressCB = Callable[[int, int], None]


def export_project(
    config: ExportConfig,
    db: Session | None = None,
    progress_callback: ProgressCB | None = None,
) -> ExportResult:
    """Dispatch to the correct exporter based on config.format."""
    dispatchers = {
        ExportFormat.YOLO: export_yolo,
        ExportFormat.COCO: export_coco,
        ExportFormat.CVAT_XML: export_cvat_xml,
        ExportFormat.PASCAL_VOC: export_pascal_voc,
        ExportFormat.CSV: export_csv,
    }
    fn = dispatchers.get(config.format)
    if fn is None:
        raise ValueError(f"Unsupported export format: {config.format}")
    return fn(config, db=db, progress_callback=progress_callback)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _get_exportable_images(
    config: ExportConfig,
    db: Session,
) -> tuple[DBProject, list[tuple[DBImage, list[Annotation]]]]:
    """Fetch project + images + annotations filtered by config."""
    proj = db.get(DBProject, config.project_id)
    if proj is None:
        raise ValueError(f"Project not found: {config.project_id}")

    images = get_images(config.project_id, limit=999_999, db=db)

    # Filter by status
    if config.only_approved:
        images = [im for im in images if im.review_decision == "approved"]
    elif config.only_reviewed:
        images = [im for im in images if im.status in ("reviewed", "annotated")]
    elif config.status_filter:
        allowed = {s.value for s in config.status_filter}
        images = [im for im in images if im.status in allowed]

    # Load annotations
    result: list[tuple[DBImage, list[Annotation]]] = []
    for img in images:
        anns = get_annotations(img.id, db)
        if not config.include_auto_annotations:
            anns = [a for a in anns if not a.is_auto]
        if config.class_filter:
            anns = [a for a in anns if a.label_name in config.class_filter]
        result.append((img, anns))

    return proj, result


def _train_val_split(
    items: list,
    ratio: float,
) -> tuple[list, list]:
    shuffled = list(items)
    random.shuffle(shuffled)
    split = int(len(shuffled) * ratio)
    return shuffled[:split], shuffled[split:]


def _class_name_to_id(proj: DBProject) -> dict[str, int]:
    """Map label class names to sequential IDs (0-based)."""
    classes = proj.label_classes  # list[dict]
    return {cls["name"]: i for i, cls in enumerate(classes)}


# ─── YOLO ────────────────────────────────────────────────────────────────────

def export_yolo(
    config: ExportConfig,
    db: Session | None = None,
    progress_callback: ProgressCB | None = None,
) -> ExportResult:
    t0 = time.time()
    close = db is None
    if db is None:
        db = get_session()

    try:
        proj, data = _get_exportable_images(config, db)
        name_to_id = _class_name_to_id(proj)
        out = Path(config.output_dir)

        # Dirs
        for split in ("train", "val"):
            (out / "images" / split).mkdir(parents=True, exist_ok=True)
            (out / "labels" / split).mkdir(parents=True, exist_ok=True)

        train_data, val_data = _train_val_split(data, config.train_ratio)
        files_created: list[str] = []
        total_anns = 0

        for split_name, split_data in [("train", train_data), ("val", val_data)]:
            for i, (img, anns) in enumerate(split_data):
                if progress_callback:
                    progress_callback(i, len(split_data))

                # Copy image
                if config.include_images:
                    dst = out / "images" / split_name / img.file_name
                    if not dst.exists():
                        shutil.copy2(img.file_path, dst)
                        files_created.append(str(dst))

                # Write label
                label_path = out / "labels" / split_name / (Path(img.file_name).stem + ".txt")
                lines: list[str] = []
                for ann in anns:
                    if ann.bbox and ann.label_name in name_to_id:
                        lines.append(bbox_to_yolo(ann.bbox, name_to_id[ann.label_name]))
                        total_anns += 1
                label_path.write_text("\n".join(lines) + "\n" if lines else "")
                files_created.append(str(label_path))

        # dataset.yaml
        class_names = [cls["name"] for cls in proj.label_classes]
        yaml_content = (
            f"path: {out.resolve()}\n"
            f"train: images/train\n"
            f"val: images/val\n"
            f"nc: {len(class_names)}\n"
            f"names: {class_names}\n"
        )
        yaml_path = out / "dataset.yaml"
        yaml_path.write_text(yaml_content)
        files_created.append(str(yaml_path))

        return ExportResult(
            format=ExportFormat.YOLO,
            output_dir=str(out),
            total_images=len(data),
            total_annotations=total_anns,
            train_images=len(train_data),
            val_images=len(val_data),
            files_created=files_created,
            export_time_seconds=time.time() - t0,
        )
    finally:
        if close:
            db.close()


# ─── COCO ────────────────────────────────────────────────────────────────────

def export_coco(
    config: ExportConfig,
    db: Session | None = None,
    progress_callback: ProgressCB | None = None,
) -> ExportResult:
    t0 = time.time()
    close = db is None
    if db is None:
        db = get_session()

    try:
        proj, data = _get_exportable_images(config, db)
        name_to_id = _class_name_to_id(proj)
        out = Path(config.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        categories = [
            {"id": i, "name": cls["name"], "supercategory": cls.get("supercategory", "")}
            for cls in proj.label_classes
            for i in [name_to_id[cls["name"]]]
        ]

        coco_images: list[dict] = []
        coco_annotations: list[dict] = []
        ann_id = 1
        total_anns = 0

        for i, (img, anns) in enumerate(data):
            if progress_callback:
                progress_callback(i, len(data))

            img_entry = {
                "id": i + 1,
                "file_name": img.file_name,
                "width": img.width,
                "height": img.height,
            }
            coco_images.append(img_entry)

            if config.include_images:
                dst = out / "images" / img.file_name
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    shutil.copy2(img.file_path, dst)

            for ann in anns:
                if ann.label_name not in name_to_id:
                    continue
                cat_id = name_to_id[ann.label_name]
                coco_ann: dict = {
                    "id": ann_id,
                    "image_id": i + 1,
                    "category_id": cat_id,
                    "iscrowd": 0,
                }
                if ann.bbox:
                    x, y, w, h = bbox_to_abs(ann.bbox, img.width, img.height)
                    coco_ann["bbox"] = [x, y, w, h]
                    coco_ann["area"] = w * h
                if ann.mask and ann.mask.rle:
                    coco_ann["segmentation"] = ann.mask.rle
                elif ann.polyline and ann.polyline.points:
                    seg = []
                    for pt in ann.polyline.points:
                        seg.extend([round(pt.x * img.width), round(pt.y * img.height)])
                    coco_ann["segmentation"] = [seg]

                coco_annotations.append(coco_ann)
                ann_id += 1
                total_anns += 1

        coco_json = {
            "info": {"description": proj.name, "date_created": datetime.utcnow().isoformat()},
            "licenses": [],
            "categories": categories,
            "images": coco_images,
            "annotations": coco_annotations,
        }
        json_path = out / "annotations.json"
        json_path.write_text(json.dumps(coco_json, indent=2))

        return ExportResult(
            format=ExportFormat.COCO,
            output_dir=str(out),
            total_images=len(data),
            total_annotations=total_anns,
            files_created=[str(json_path)],
            export_time_seconds=time.time() - t0,
        )
    finally:
        if close:
            db.close()


# ─── CVAT XML ────────────────────────────────────────────────────────────────

def export_cvat_xml(
    config: ExportConfig,
    db: Session | None = None,
    progress_callback: ProgressCB | None = None,
) -> ExportResult:
    t0 = time.time()
    close = db is None
    if db is None:
        db = get_session()

    try:
        proj, data = _get_exportable_images(config, db)
        out = Path(config.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        root = ET.Element("annotations")
        ET.SubElement(root, "version").text = "1.1"

        meta = ET.SubElement(root, "meta")
        task = ET.SubElement(meta, "task")
        ET.SubElement(task, "name").text = proj.name
        ET.SubElement(task, "size").text = str(len(data))

        labels_el = ET.SubElement(task, "labels")
        for cls in proj.label_classes:
            label_el = ET.SubElement(labels_el, "label")
            ET.SubElement(label_el, "name").text = cls["name"]
            ET.SubElement(label_el, "color").text = cls.get("color", "#000000")

        total_anns = 0
        for i, (img, anns) in enumerate(data):
            if progress_callback:
                progress_callback(i, len(data))

            img_el = ET.SubElement(root, "image")
            img_el.set("id", str(i))
            img_el.set("name", img.file_name)
            img_el.set("width", str(img.width))
            img_el.set("height", str(img.height))

            for ann in anns:
                if ann.bbox:
                    x, y, w, h = bbox_to_abs(ann.bbox, img.width, img.height)
                    box_el = ET.SubElement(img_el, "box")
                    box_el.set("label", ann.label_name)
                    box_el.set("xtl", str(x))
                    box_el.set("ytl", str(y))
                    box_el.set("xbr", str(x + w))
                    box_el.set("ybr", str(y + h))
                    if ann.confidence is not None:
                        box_el.set("score", f"{ann.confidence:.3f}")
                    total_anns += 1

                elif ann.polyline and ann.polyline.points:
                    tag = "polygon" if ann.polyline.is_closed else "polyline"
                    poly_el = ET.SubElement(img_el, tag)
                    poly_el.set("label", ann.label_name)
                    pts = ";".join(
                        f"{round(p.x * img.width)},{round(p.y * img.height)}"
                        for p in ann.polyline.points
                    )
                    poly_el.set("points", pts)
                    total_anns += 1

        xml_str = minidom.parseString(ET.tostring(root, encoding="unicode")).toprettyxml(indent="  ")
        xml_path = out / "annotations.xml"
        xml_path.write_text(xml_str, encoding="utf-8")

        return ExportResult(
            format=ExportFormat.CVAT_XML,
            output_dir=str(out),
            total_images=len(data),
            total_annotations=total_anns,
            files_created=[str(xml_path)],
            export_time_seconds=time.time() - t0,
        )
    finally:
        if close:
            db.close()


# ─── Pascal VOC ──────────────────────────────────────────────────────────────

def export_pascal_voc(
    config: ExportConfig,
    db: Session | None = None,
    progress_callback: ProgressCB | None = None,
) -> ExportResult:
    t0 = time.time()
    close = db is None
    if db is None:
        db = get_session()

    try:
        proj, data = _get_exportable_images(config, db)
        out = Path(config.output_dir)
        (out / "Annotations").mkdir(parents=True, exist_ok=True)
        if config.include_images:
            (out / "JPEGImages").mkdir(parents=True, exist_ok=True)

        files_created: list[str] = []
        total_anns = 0

        for i, (img, anns) in enumerate(data):
            if progress_callback:
                progress_callback(i, len(data))

            if config.include_images:
                dst = out / "JPEGImages" / img.file_name
                if not dst.exists():
                    shutil.copy2(img.file_path, dst)

            root = ET.Element("annotation")
            ET.SubElement(root, "folder").text = "JPEGImages"
            ET.SubElement(root, "filename").text = img.file_name
            size_el = ET.SubElement(root, "size")
            ET.SubElement(size_el, "width").text = str(img.width)
            ET.SubElement(size_el, "height").text = str(img.height)
            ET.SubElement(size_el, "depth").text = "3"

            for ann in anns:
                if ann.bbox:
                    x, y, w, h = bbox_to_abs(ann.bbox, img.width, img.height)
                    obj = ET.SubElement(root, "object")
                    ET.SubElement(obj, "name").text = ann.label_name
                    ET.SubElement(obj, "difficult").text = "0"
                    bndbox = ET.SubElement(obj, "bndbox")
                    ET.SubElement(bndbox, "xmin").text = str(x)
                    ET.SubElement(bndbox, "ymin").text = str(y)
                    ET.SubElement(bndbox, "xmax").text = str(x + w)
                    ET.SubElement(bndbox, "ymax").text = str(y + h)
                    total_anns += 1

            xml_path = out / "Annotations" / (Path(img.file_name).stem + ".xml")
            xml_str = minidom.parseString(ET.tostring(root, encoding="unicode")).toprettyxml(indent="  ")
            xml_path.write_text(xml_str, encoding="utf-8")
            files_created.append(str(xml_path))

        return ExportResult(
            format=ExportFormat.PASCAL_VOC,
            output_dir=str(out),
            total_images=len(data),
            total_annotations=total_anns,
            files_created=files_created,
            export_time_seconds=time.time() - t0,
        )
    finally:
        if close:
            db.close()


# ─── CSV ─────────────────────────────────────────────────────────────────────

def export_csv(
    config: ExportConfig,
    db: Session | None = None,
    progress_callback: ProgressCB | None = None,
) -> ExportResult:
    t0 = time.time()
    close = db is None
    if db is None:
        db = get_session()

    try:
        proj, data = _get_exportable_images(config, db)
        out = Path(config.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        csv_path = out / "annotations.csv"
        total_anns = 0

        with open(csv_path, "w") as f:
            f.write("image_id,image_path,label,annotation_type,x,y,width,height,confidence,is_auto,track_id,annotator,created_at\n")
            for i, (img, anns) in enumerate(data):
                if progress_callback:
                    progress_callback(i, len(data))
                for ann in anns:
                    x = y = w = h = ""
                    if ann.bbox:
                        x, y, w, h = (
                            f"{ann.bbox.x:.6f}",
                            f"{ann.bbox.y:.6f}",
                            f"{ann.bbox.width:.6f}",
                            f"{ann.bbox.height:.6f}",
                        )
                    conf = f"{ann.confidence:.3f}" if ann.confidence is not None else ""
                    track = str(ann.track_id) if ann.track_id is not None else ""
                    f.write(
                        f"{img.id},{img.file_path},{ann.label_name},{ann.annotation_type.value},"
                        f"{x},{y},{w},{h},{conf},{ann.is_auto},{track},{ann.created_by},"
                        f"{ann.created_at.isoformat() if ann.created_at else ''}\n"
                    )
                    total_anns += 1

        return ExportResult(
            format=ExportFormat.CSV,
            output_dir=str(out),
            total_images=len(data),
            total_annotations=total_anns,
            files_created=[str(csv_path)],
            export_time_seconds=time.time() - t0,
        )
    finally:
        if close:
            db.close()
