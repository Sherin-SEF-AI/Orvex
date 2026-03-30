"""
labelox/core/importer.py — Import annotations from YOLO, COCO, CVAT XML, VOC formats.
"""
from __future__ import annotations

import json
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

from loguru import logger
from sqlalchemy.orm import Session

from labelox.core.annotation_engine import save_annotations
from labelox.core.database import DBImage, get_session
from labelox.core.models import (
    Annotation,
    AnnotationType,
    BBoxAnnotation,
    MaskAnnotation,
    Point,
    PolylineAnnotation,
)


# ─── YOLO ────────────────────────────────────────────────────────────────────

def import_yolo(
    label_dir: str | Path,
    image_dir: str | Path,
    classes_file: str | Path | None = None,
    class_names: list[str] | None = None,
    project_id: str | None = None,
    annotator_name: str = "import",
    db: Session | None = None,
) -> int:
    """Import YOLO format labels. Returns count of annotations imported.

    Either pass classes_file (path to classes.txt or data.yaml) or class_names list.
    """
    label_dir = Path(label_dir)
    image_dir = Path(image_dir)

    # Resolve class names
    names: list[str] = []
    if class_names:
        names = class_names
    elif classes_file:
        cf = Path(classes_file)
        if cf.suffix in (".yaml", ".yml"):
            import yaml
            with open(cf) as f:
                data = yaml.safe_load(f)
            names = data.get("names", [])
        else:
            names = [line.strip() for line in cf.read_text().splitlines() if line.strip()]

    if not names:
        raise ValueError("No class names provided. Pass classes_file or class_names.")

    close = db is None
    if db is None:
        db = get_session()

    count = 0
    try:
        for txt_file in sorted(label_dir.glob("*.txt")):
            stem = txt_file.stem
            # Find matching image
            img_record = _find_image_by_stem(stem, image_dir, project_id, db)
            if img_record is None:
                logger.warning("No image match for label: {}", txt_file.name)
                continue

            anns: list[Annotation] = []
            for line in txt_file.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                cls_id = int(parts[0])
                if cls_id >= len(names):
                    continue
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                anns.append(Annotation(
                    image_id=img_record.id,
                    label_id=str(cls_id),
                    label_name=names[cls_id],
                    annotation_type=AnnotationType.BBOX,
                    bbox=BBoxAnnotation(x=cx - w / 2, y=cy - h / 2, width=w, height=h),
                    created_by=annotator_name,
                ))
                count += 1

            if anns:
                save_annotations(img_record.id, anns, annotator_name, db)

        logger.info("Imported {} YOLO annotations", count)
        return count
    finally:
        if close:
            db.close()


# ─── COCO ────────────────────────────────────────────────────────────────────

def import_coco(
    json_path: str | Path,
    project_id: str | None = None,
    annotator_name: str = "import",
    db: Session | None = None,
) -> int:
    """Import COCO JSON annotations. Returns count imported."""
    with open(json_path) as f:
        coco = json.load(f)

    cats = {c["id"]: c["name"] for c in coco.get("categories", [])}
    imgs = {i["id"]: i for i in coco.get("images", [])}

    close = db is None
    if db is None:
        db = get_session()

    count = 0
    # Group annotations by image
    by_image: dict[int, list[dict]] = {}
    for ann in coco.get("annotations", []):
        by_image.setdefault(ann["image_id"], []).append(ann)

    try:
        for img_id, ann_list in by_image.items():
            img_info = imgs.get(img_id)
            if not img_info:
                continue

            img_record = _find_image_by_name(img_info["file_name"], project_id, db)
            if img_record is None:
                continue

            w, h = img_record.width, img_record.height
            anns: list[Annotation] = []

            for coco_ann in ann_list:
                cat_name = cats.get(coco_ann["category_id"], "unknown")
                ann_kwargs: dict = {
                    "image_id": img_record.id,
                    "label_id": str(coco_ann["category_id"]),
                    "label_name": cat_name,
                    "created_by": annotator_name,
                }

                # BBox
                if "bbox" in coco_ann and coco_ann["bbox"]:
                    bx, by, bw, bh = coco_ann["bbox"]
                    ann_kwargs["annotation_type"] = AnnotationType.BBOX
                    ann_kwargs["bbox"] = BBoxAnnotation(
                        x=bx / w, y=by / h, width=bw / w, height=bh / h,
                    )

                # Segmentation (polygon)
                if "segmentation" in coco_ann and isinstance(coco_ann["segmentation"], list):
                    segs = coco_ann["segmentation"]
                    if segs and isinstance(segs[0], list):
                        pts = []
                        coords = segs[0]
                        for j in range(0, len(coords), 2):
                            pts.append(Point(x=coords[j] / w, y=coords[j + 1] / h))
                        if pts:
                            ann_kwargs["annotation_type"] = AnnotationType.POLYGON
                            ann_kwargs["polyline"] = PolylineAnnotation(points=pts, is_closed=True)

                # RLE mask
                elif "segmentation" in coco_ann and isinstance(coco_ann["segmentation"], dict):
                    ann_kwargs["annotation_type"] = AnnotationType.MASK
                    ann_kwargs["mask"] = MaskAnnotation(rle=coco_ann["segmentation"])

                if "annotation_type" not in ann_kwargs:
                    ann_kwargs["annotation_type"] = AnnotationType.BBOX

                anns.append(Annotation(**ann_kwargs))
                count += 1

            if anns:
                save_annotations(img_record.id, anns, annotator_name, db)

        logger.info("Imported {} COCO annotations", count)
        return count
    finally:
        if close:
            db.close()


# ─── CVAT XML ────────────────────────────────────────────────────────────────

def import_cvat_xml(
    xml_path: str | Path,
    project_id: str | None = None,
    annotator_name: str = "import",
    db: Session | None = None,
) -> int:
    """Import CVAT XML 1.1 annotations. Returns count imported."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    close = db is None
    if db is None:
        db = get_session()

    count = 0
    try:
        for img_el in root.findall("image"):
            name = img_el.get("name", "")
            w = int(img_el.get("width", "0"))
            h = int(img_el.get("height", "0"))

            img_record = _find_image_by_name(name, project_id, db)
            if img_record is None:
                continue
            if w == 0:
                w = img_record.width
            if h == 0:
                h = img_record.height

            anns: list[Annotation] = []

            # Boxes
            for box_el in img_el.findall("box"):
                label = box_el.get("label", "")
                xtl = float(box_el.get("xtl", "0"))
                ytl = float(box_el.get("ytl", "0"))
                xbr = float(box_el.get("xbr", "0"))
                ybr = float(box_el.get("ybr", "0"))
                conf = box_el.get("score")

                anns.append(Annotation(
                    image_id=img_record.id,
                    label_id=label,
                    label_name=label,
                    annotation_type=AnnotationType.BBOX,
                    bbox=BBoxAnnotation(
                        x=xtl / w, y=ytl / h,
                        width=(xbr - xtl) / w, height=(ybr - ytl) / h,
                    ),
                    confidence=float(conf) if conf else None,
                    created_by=annotator_name,
                ))
                count += 1

            # Polygons
            for poly_el in img_el.findall("polygon"):
                label = poly_el.get("label", "")
                pts_str = poly_el.get("points", "")
                pts = []
                for pair in pts_str.split(";"):
                    xy = pair.strip().split(",")
                    if len(xy) == 2:
                        pts.append(Point(x=float(xy[0]) / w, y=float(xy[1]) / h))
                if pts:
                    anns.append(Annotation(
                        image_id=img_record.id,
                        label_id=label,
                        label_name=label,
                        annotation_type=AnnotationType.POLYGON,
                        polyline=PolylineAnnotation(points=pts, is_closed=True),
                        created_by=annotator_name,
                    ))
                    count += 1

            # Polylines
            for line_el in img_el.findall("polyline"):
                label = line_el.get("label", "")
                pts_str = line_el.get("points", "")
                pts = []
                for pair in pts_str.split(";"):
                    xy = pair.strip().split(",")
                    if len(xy) == 2:
                        pts.append(Point(x=float(xy[0]) / w, y=float(xy[1]) / h))
                if pts:
                    anns.append(Annotation(
                        image_id=img_record.id,
                        label_id=label,
                        label_name=label,
                        annotation_type=AnnotationType.POLYLINE,
                        polyline=PolylineAnnotation(points=pts, is_closed=False),
                        created_by=annotator_name,
                    ))
                    count += 1

            if anns:
                save_annotations(img_record.id, anns, annotator_name, db)

        logger.info("Imported {} CVAT annotations", count)
        return count
    finally:
        if close:
            db.close()


# ─── Pascal VOC ──────────────────────────────────────────────────────────────

def import_pascal_voc(
    xml_dir: str | Path,
    project_id: str | None = None,
    annotator_name: str = "import",
    db: Session | None = None,
) -> int:
    """Import Pascal VOC XML annotations. Returns count imported."""
    xml_dir = Path(xml_dir)

    close = db is None
    if db is None:
        db = get_session()

    count = 0
    try:
        for xml_file in sorted(xml_dir.glob("*.xml")):
            tree = ET.parse(xml_file)
            root = tree.getroot()

            fname_el = root.find("filename")
            size_el = root.find("size")
            if fname_el is None or size_el is None:
                continue

            fname = fname_el.text or ""
            w = int(size_el.findtext("width", "0"))
            h = int(size_el.findtext("height", "0"))

            img_record = _find_image_by_name(fname, project_id, db)
            if img_record is None:
                continue
            if w == 0:
                w = img_record.width
            if h == 0:
                h = img_record.height

            anns: list[Annotation] = []
            for obj in root.findall("object"):
                name = obj.findtext("name", "")
                bndbox = obj.find("bndbox")
                if bndbox is None:
                    continue
                xmin = float(bndbox.findtext("xmin", "0"))
                ymin = float(bndbox.findtext("ymin", "0"))
                xmax = float(bndbox.findtext("xmax", "0"))
                ymax = float(bndbox.findtext("ymax", "0"))

                anns.append(Annotation(
                    image_id=img_record.id,
                    label_id=name,
                    label_name=name,
                    annotation_type=AnnotationType.BBOX,
                    bbox=BBoxAnnotation(
                        x=xmin / w, y=ymin / h,
                        width=(xmax - xmin) / w, height=(ymax - ymin) / h,
                    ),
                    created_by=annotator_name,
                ))
                count += 1

            if anns:
                save_annotations(img_record.id, anns, annotator_name, db)

        logger.info("Imported {} VOC annotations", count)
        return count
    finally:
        if close:
            db.close()


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _find_image_by_stem(
    stem: str,
    image_dir: Path,
    project_id: str | None,
    db: Session,
) -> DBImage | None:
    """Find a DB image record matching a filename stem."""
    q = db.query(DBImage).filter(DBImage.file_name.like(f"{stem}.%"))
    if project_id:
        q = q.filter(DBImage.project_id == project_id)
    return q.first()


def _find_image_by_name(
    file_name: str,
    project_id: str | None,
    db: Session,
) -> DBImage | None:
    """Find a DB image record by exact filename."""
    q = db.query(DBImage).filter(DBImage.file_name == file_name)
    if project_id:
        q = q.filter(DBImage.project_id == project_id)
    return q.first()
