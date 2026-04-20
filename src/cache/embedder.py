"""
Local sentence-transformers embedder.
Uses all-MiniLM-L6-v2 (22MB, fast, good semantic quality).
Model is loaded once at process start and reused.
"""

from __future__ import annotations

import numpy as np
import structlog
from sentence_transformers import SentenceTransformer

log = structlog.get_logger()

_MODEL_NAME = "all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        log.info("embedder.loading", model=_MODEL_NAME)
        _model = SentenceTransformer(_MODEL_NAME)
        log.info("embedder.ready", model=_MODEL_NAME)
    return _model


def embed(text: str) -> list[float]:
    """Return a normalised embedding vector for `text`."""
    model = _get_model()
    vec: np.ndarray = model.encode(text, normalize_embeddings=True)
    return vec.tolist()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Cosine similarity between two pre-normalised vectors.
    Since both are L2-normalised, this is just the dot product.
    """
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    return float(np.dot(va, vb))
