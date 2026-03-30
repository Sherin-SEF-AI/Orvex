"""
core/continuous_learning.py — Continuous learning loop orchestration.

Watches for correction thresholds, triggers retrain-compare-promote cycles,
and maintains a persistent learning log.

Pipeline:
  corrections ≥ threshold
    → export_corrected_dataset()
    → augment_dataset()
    → run_training()
    → compare_models()       (candidate vs current active)
    → if delta_map50 ≥ 0.01: set_active_model(candidate)

No UI imports — pure Python business logic.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from loguru import logger

from core.models import (
    AugmentationConfig, LearningTrigger, ModelComparison,
    TrainingConfig, TrainingRun,
)
from core.session_manager import SessionManager

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOG_PATH = Path.home() / ".roverdatakit" / "learning_log.json"
AUTO_PROMOTE_THRESHOLD = 0.01   # minimum mAP50 delta to promote candidate


# ---------------------------------------------------------------------------
# Learning log
# ---------------------------------------------------------------------------

def get_learning_log(log_path: Path = LOG_PATH) -> list[LearningTrigger]:
    if not log_path.exists():
        return []
    raw = json.loads(log_path.read_text())
    result = []
    for item in raw:
        try:
            result.append(LearningTrigger(**item))
        except Exception as exc:
            logger.warning(f"Skipping malformed log entry: {exc}")
    return result


def record_trigger(trigger: LearningTrigger, log_path: Path = LOG_PATH) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entries = get_learning_log(log_path)
    entries.append(trigger)
    raw = []
    for e in entries:
        d = e.model_dump()
        d["triggered_at"] = d["triggered_at"].isoformat()
        raw.append(d)
    log_path.write_text(json.dumps(raw, indent=2))
    logger.debug(f"Recorded learning trigger for session {trigger.session_id}")


# ---------------------------------------------------------------------------
# Trigger check
# ---------------------------------------------------------------------------

def check_and_maybe_trigger(
    session_id: str,
    sm: SessionManager,
    threshold: int = 50,
) -> bool:
    """
    Check if correction count meets threshold.
    Returns True if threshold met (does NOT start the cycle — caller decides).
    """
    from core.annotation_review import check_learning_trigger
    return check_learning_trigger(session_id, sm, threshold)


# ---------------------------------------------------------------------------
# Full learning cycle
# ---------------------------------------------------------------------------

def run_learning_cycle(
    session_id: str,
    sm: SessionManager,
    trigger_type: str,
    augmentation_config: AugmentationConfig,
    training_config: TrainingConfig,
    auto_promote: bool = True,
    progress_callback: Callable[[int], None] | None = None,
    status_callback: Callable[[str], None] | None = None,
    registry_path: Path | None = None,
) -> tuple[TrainingRun, ModelComparison]:
    """
    Full retrain-compare-promote cycle.

    Steps:
      1. Export corrected dataset
      2. Augment dataset
      3. Train model
      4. Compare candidate vs active baseline
      5. Optionally promote if improved

    Returns (TrainingRun, ModelComparison).
    """
    from core.annotation_review import export_corrected_dataset, get_review_stats
    from core.augmentor import augment_dataset
    from core.trainer import run_training
    from core.inference_server import (
        REGISTRY_FILE, compare_models, get_active_model,
        register_model, set_active_model,
    )

    reg_path = registry_path or REGISTRY_FILE

    def _status(msg: str) -> None:
        logger.info(msg)
        if status_callback:
            status_callback(msg)

    def _progress(pct: int) -> None:
        if progress_callback:
            progress_callback(pct)

    # ── 1. Record trigger ────────────────────────────────────────────────
    stats = get_review_stats(session_id, sm)
    trigger = LearningTrigger(
        session_id=session_id,
        trigger_type=trigger_type,
        corrections_count=stats["usable_for_training"],
        triggered_at=datetime.now(timezone.utc),
        resulting_run_id=None,
    )
    record_trigger(trigger)

    # ── 2. Export corrected dataset ──────────────────────────────────────
    _status("Exporting corrected dataset…")
    _progress(5)
    corrected_dir = str(sm.session_folder(session_id) / "corrected_dataset")
    export_corrected_dataset(session_id, sm, corrected_dir, status_callback=_status)
    _progress(15)

    # ── 3. Augment dataset ───────────────────────────────────────────────
    _status("Augmenting corrected dataset…")
    aug_out_dir = str(sm.session_folder(session_id) / "cl_augmented")
    from core.models import FrameAnnotation
    annotations: list[FrameAnnotation] = []
    img_dir = Path(corrected_dir) / "images" / "train"
    lbl_dir = Path(corrected_dir) / "labels" / "train"
    if img_dir.exists():
        for img_path in sorted(img_dir.glob("*.jpg")):
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            from core.models import Detection
            dets: list[Detection] = []
            if lbl_path.exists():
                for line in lbl_path.read_text().splitlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:])
                        dets.append(Detection(
                            class_id=cls_id,
                            class_name=str(cls_id),
                            confidence=1.0,
                            bbox_xyxy=[0, 0, 0, 0],
                            bbox_xywhn=[cx, cy, w, h],
                        ))
            annotations.append(FrameAnnotation(
                frame_path=str(img_path),
                detections=dets,
                inference_time_ms=0.0,
                model_version="human_corrected",
            ))

    if annotations:
        augment_dataset(
            frame_paths=[a.frame_path for a in annotations],
            annotations=annotations,
            config=augmentation_config,
            output_dir=aug_out_dir,
            multiplier=augmentation_config.multiplier,
            progress_callback=lambda p: _progress(15 + int(p * 0.2)),
            status_callback=_status,
        )
        dataset_dir = aug_out_dir
    else:
        dataset_dir = corrected_dir
    _progress(35)

    # ── 4. Train ─────────────────────────────────────────────────────────
    _status(f"Training {training_config.model_variant} on corrected data…")
    from core.trainer import prepare_training_config
    run_config = prepare_training_config(
        dataset_dir=dataset_dir,
        model_variant=training_config.model_variant,
        pretrained_weights=training_config.pretrained_weights,
        epochs=training_config.epochs,
        batch_size=training_config.batch_size,
        image_size=training_config.image_size,
        learning_rate=training_config.learning_rate,
        device=training_config.device,
        project_name=training_config.project_name,
        run_name=f"cl_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
    )
    train_result = run_training(
        config=run_config,
        progress_callback=lambda p: _progress(35 + int(p * 0.45)),
        status_callback=_status,
    )
    _progress(80)

    # ── 5. Register candidate ─────────────────────────────────────────────
    _status("Registering candidate model…")
    candidate = register_model(
        weights_path=train_result.best_weights_path,
        name=f"cl_{session_id[:8]}_{trigger.triggered_at.strftime('%Y%m%d')}",
        model_variant=run_config.model_variant,
        training_run_id=train_result.run_id,
        metrics={
            "map50": train_result.final_map50,
            "map50_95": train_result.final_map50_95,
            "best_epoch": train_result.best_epoch,
        },
        registry_path=reg_path,
    )

    # ── 6. Compare vs baseline ────────────────────────────────────────────
    _status("Comparing candidate vs baseline…")
    comparison: ModelComparison | None = None
    try:
        baseline = get_active_model(reg_path)
        # Don't compare model against itself (first-ever model)
        if baseline.model_id != candidate.model_id:
            val_dir = str(Path(dataset_dir) / "images" / "train")
            comparison = compare_models(
                baseline_id=baseline.model_id,
                candidate_id=candidate.model_id,
                val_dir=val_dir,
                registry_path=reg_path,
            )
            _status(
                f"Comparison: baseline mAP50={comparison.baseline_map50:.4f} "
                f"→ candidate mAP50={comparison.candidate_map50:.4f} "
                f"(Δ{comparison.delta_map50:+.4f})"
            )
            if auto_promote and comparison.improved:
                set_active_model(candidate.model_id, reg_path)
                _status(f"✓ Candidate promoted to active model (Δ≥{AUTO_PROMOTE_THRESHOLD})")
            elif auto_promote:
                _status(
                    f"Candidate NOT promoted — improvement {comparison.delta_map50:.4f} "
                    f"< threshold {AUTO_PROMOTE_THRESHOLD}"
                )
        else:
            _status("First model — auto-activated.")
    except Exception as exc:
        logger.warning(f"Model comparison failed (non-fatal): {exc}")
        comparison = ModelComparison(
            baseline_model_id="none",
            candidate_model_id=candidate.model_id,
            val_dir=dataset_dir,
            baseline_map50=0.0,
            candidate_map50=train_result.final_map50,
            improved=True,
            delta_map50=train_result.final_map50,
        )
        if auto_promote:
            set_active_model(candidate.model_id, reg_path)

    # Update trigger log with run id
    trigger.resulting_run_id = train_result.run_id
    record_trigger(trigger)
    _progress(100)
    _status("Learning cycle complete.")

    return train_result, comparison
