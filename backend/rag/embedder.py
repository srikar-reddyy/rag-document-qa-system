"""
Embedding Generator.

Uses sentence-transformers when available and falls back to a deterministic
hash-based embedding so the backend can run without heavyweight model deps.
"""

from typing import List, Union
import hashlib
import logging
import math
import re

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

logger = logging.getLogger(__name__)

_embedding_model = None
MODEL_NAME = "all-MiniLM-L6-v2"
FALLBACK_DIMENSION = 384


class _FallbackEmbeddingModel:
    def __init__(self, dimension: int = FALLBACK_DIMENSION):
        self._dimension = dimension

    def get_sentence_embedding_dimension(self) -> int:
        return self._dimension

    def encode(self, texts, convert_to_numpy: bool = False, show_progress_bar: bool = False):
        if isinstance(texts, str):
            return self._encode_single(texts)
        return [self._encode_single(text) for text in texts]

    def _encode_single(self, text: str) -> List[float]:
        vector = [0.0] * self._dimension
        tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest, "big") % self._dimension
            vector[bucket] += 1.0

        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


def get_embedding_model():
    """
    Get or initialize the embedding model (singleton pattern).
    """
    global _embedding_model

    if _embedding_model is None:
        if SentenceTransformer is None:
            logger.warning("sentence-transformers is unavailable; using fallback embeddings")
            _embedding_model = _FallbackEmbeddingModel()
        else:
            logger.info(f"Loading embedding model: {MODEL_NAME}")
            _embedding_model = SentenceTransformer(MODEL_NAME)
            logger.info("Embedding model loaded successfully")

    return _embedding_model


def _to_list(embedding):
    if hasattr(embedding, "tolist"):
        return embedding.tolist()
    return embedding


def embed_text(text: str) -> List[float]:
    """Generate embedding for a single text."""
    model = get_embedding_model()
    embedding = model.encode(text, convert_to_numpy=True)
    return _to_list(embedding)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for multiple texts (batch processing)."""
    model = get_embedding_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    return _to_list(embeddings)


def get_embedding_dimension() -> int:
    """Get the dimension of embeddings from the model."""
    model = get_embedding_model()
    return model.get_sentence_embedding_dimension()
