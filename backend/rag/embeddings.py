"""Sentence embedding service using HuggingFace Sentence-Transformers."""

from __future__ import annotations

import threading
from typing import List, Optional

import numpy as np

from backend.config import settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)

_lock = threading.Lock()


class EmbeddingService:
    """Singleton wrapper for a SentenceTransformer model."""

    _instance: Optional["EmbeddingService"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with _lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._model = None
                    cls._instance._model_name = None
        return cls._instance

    def __init__(self, model_name: str | None = None) -> None:
        self.requested_model_name = model_name or settings.embedding_model_name

    def _get_device(self) -> str:
        if not settings.use_gpu_if_available:
            return "cpu"

        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._model_name == self.requested_model_name:
            return

        with _lock:
            if self._model is not None and self._model_name == self.requested_model_name:
                return

            from sentence_transformers import SentenceTransformer

            device = self._get_device()

            logger.info(
                "Loading embedding model '%s' on '%s'",
                self.requested_model_name,
                device,
            )

            self._model = SentenceTransformer(
                self.requested_model_name,
                device=device,
            )
            self._model_name = self.requested_model_name

    @property
    def dimension(self) -> int:
        self._ensure_loaded()
        return int(self._model.get_sentence_embedding_dimension())

    def embed(
        self,
        texts: List[str],
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Generate normalized embeddings for a list of texts."""

        if not texts:
            return np.zeros((0, self.dimension), dtype="float32")

        self._ensure_loaded()

        vectors = self._model.encode(
            texts,
            batch_size=batch_size or settings.embedding_batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return vectors.astype("float32")

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()