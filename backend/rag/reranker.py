"""Cross-encoder reranking for retrieved document candidates."""

from __future__ import annotations

import threading
from typing import List, Optional, Tuple

from backend.config import settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)
_lock = threading.Lock()


class CrossEncoderReranker:
    _instance: Optional["CrossEncoderReranker"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with _lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._model = None
        return cls._instance

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        with _lock:
            if self._model is not None:
                return

            from sentence_transformers import CrossEncoder

            logger.info(
                "Loading cross-encoder model '%s'",
                settings.cross_encoder_model,
            )

            self._model = CrossEncoder(settings.cross_encoder_model)

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, str]],
        top_k: int,
    ) -> List[Tuple[str, str, float]]:
        if not candidates:
            return []

        self._ensure_loaded()

        pairs = [(query, text) for _, text in candidates]
        scores = self._model.predict(pairs)

        scored = [
            (chunk_id, text, float(score))
            for (chunk_id, text), score in zip(candidates, scores)
        ]

        scored.sort(key=lambda x: x[2], reverse=True)

        return scored[:top_k]


def get_reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker()