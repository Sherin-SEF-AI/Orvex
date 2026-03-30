"""
labelox/core/similarity_engine.py — Image similarity via CLIP embeddings + FAISS.

Falls back to sklearn NearestNeighbors if FAISS is not installed.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Callable

import numpy as np
from loguru import logger

ProgressCB = Callable[[int, int], None]

# ─── FAISS availability ─────────────────────────────────────────────────────

try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False
    logger.debug("FAISS not available — using sklearn fallback for similarity search")


# ─── Index Wrapper ───────────────────────────────────────────────────────────

class SimilarityIndex:
    """Wraps FAISS or sklearn for nearest-neighbor search."""

    def __init__(self, dimension: int = 512) -> None:
        self.dimension = dimension
        self.embeddings: np.ndarray | None = None
        self.image_ids: list[str] = []
        self._index: Any = None

    def build(self, embeddings: np.ndarray, image_ids: list[str]) -> None:
        """Build index from embeddings matrix [N, D]."""
        self.embeddings = embeddings.astype(np.float32)
        self.image_ids = image_ids

        if _HAS_FAISS:
            self._index = faiss.IndexFlatL2(self.dimension)
            self._index.add(self.embeddings)
            logger.info("Built FAISS index with {} vectors", len(image_ids))
        else:
            from sklearn.neighbors import NearestNeighbors
            self._index = NearestNeighbors(n_neighbors=min(20, len(image_ids)), metric="cosine")
            self._index.fit(self.embeddings)
            logger.info("Built sklearn index with {} vectors", len(image_ids))

    def search(self, query: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
        """Find top_k most similar images. Returns [(image_id, distance), ...]."""
        if self._index is None:
            return []

        query = query.astype(np.float32).reshape(1, -1)

        if _HAS_FAISS:
            distances, indices = self._index.search(query, min(top_k, len(self.image_ids)))
            results = []
            for d, i in zip(distances[0], indices[0]):
                if i >= 0:
                    results.append((self.image_ids[i], float(d)))
            return results
        else:
            distances, indices = self._index.kneighbors(query, n_neighbors=min(top_k, len(self.image_ids)))
            results = []
            for d, i in zip(distances[0], indices[0]):
                results.append((self.image_ids[i], float(d)))
            return results

    def save(self, path: str | Path) -> None:
        """Save index to disk."""
        data = {
            "embeddings": self.embeddings,
            "image_ids": self.image_ids,
            "dimension": self.dimension,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str | Path) -> None:
        """Load index from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.dimension = data["dimension"]
        self.build(data["embeddings"], data["image_ids"])


# ─── High-Level API ──────────────────────────────────────────────────────────

def build_embedding_index(
    project_id: str,
    db=None,
    model_name: str = "openai/clip-vit-base-patch32",
    device: str = "auto",
    progress_callback: ProgressCB | None = None,
) -> SimilarityIndex:
    """Build FAISS index for all images in a project."""
    from labelox.core.classifier import compute_image_embedding, load_clip
    from labelox.core.database import DBImage, get_session

    close = db is None
    if db is None:
        db = get_session()

    try:
        images = list(
            db.query(DBImage)
            .filter(DBImage.project_id == project_id)
            .order_by(DBImage.file_name)
            .all()
        )

        if not images:
            idx = SimilarityIndex()
            idx.build(np.zeros((0, 512), dtype=np.float32), [])
            return idx

        model, processor = load_clip(model_name, device)
        embeddings = []
        image_ids = []
        n = len(images)

        for i, img in enumerate(images):
            try:
                emb = compute_image_embedding(
                    img.file_path, model=model, processor=processor,
                    model_name=model_name, device=device,
                )
                embeddings.append(emb)
                image_ids.append(img.id)
            except Exception as exc:
                logger.warning("Embedding failed for {}: {}", img.file_name, exc)

            if progress_callback:
                progress_callback(i + 1, n)

        if not embeddings:
            idx = SimilarityIndex()
            idx.build(np.zeros((0, 512), dtype=np.float32), [])
            return idx

        matrix = np.stack(embeddings)
        idx = SimilarityIndex(dimension=matrix.shape[1])
        idx.build(matrix, image_ids)
        return idx

    finally:
        if close:
            db.close()


def find_similar_images(
    query_image_id: str,
    index: SimilarityIndex,
    db=None,
    top_k: int = 5,
    only_annotated: bool = True,
) -> list[tuple[str, float]]:
    """Find top_k most similar images to query.

    Returns [(image_id, similarity_score), ...]
    """
    from labelox.core.database import DBImage, get_session

    # Get query embedding
    if index.embeddings is None or query_image_id not in index.image_ids:
        return []

    idx = index.image_ids.index(query_image_id)
    query_emb = index.embeddings[idx]

    # Search
    candidates = index.search(query_emb, top_k + 1)  # +1 to exclude self

    # Filter out self and optionally only annotated
    results = []
    close = db is None
    if db is None:
        db = get_session()

    try:
        for img_id, dist in candidates:
            if img_id == query_image_id:
                continue
            if only_annotated:
                img = db.get(DBImage, img_id)
                if img and img.status not in ("annotated", "reviewed"):
                    continue
            # Convert L2 distance to similarity (higher = more similar)
            similarity = 1.0 / (1.0 + dist)
            results.append((img_id, similarity))
            if len(results) >= top_k:
                break

        return results
    finally:
        if close:
            db.close()


def suggest_labels_from_similar(
    query_image_id: str,
    similar_images: list[tuple[str, float]],
    db=None,
) -> list:
    """Aggregate annotations from similar images as suggestions."""
    from labelox.core.annotation_engine import get_annotations
    from labelox.core.database import get_session

    close = db is None
    if db is None:
        db = get_session()

    try:
        suggestions = []
        for img_id, similarity in similar_images:
            anns = get_annotations(img_id, db)
            for ann in anns:
                suggestions.append({
                    "annotation": ann,
                    "source_image_id": img_id,
                    "similarity": similarity,
                })
        # Sort by similarity
        suggestions.sort(key=lambda s: s["similarity"], reverse=True)
        return suggestions
    finally:
        if close:
            db.close()
