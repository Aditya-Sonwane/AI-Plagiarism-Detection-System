"""BM25 lexical index used for hybrid retrieval."""
from __future__ import annotations

import re
import threading
from typing import List, Optional

from rank_bm25 import BM25Okapi

from backend.utils.logger import get_logger

logger = get_logger(__name__)

_lock = threading.Lock()
_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Store:
    """Singleton BM25 index over all chunk texts."""

    _instance: Optional["BM25Store"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with _lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._dirty = True
                    cls._instance._bm25 = None
                    cls._instance._chunk_ids = []
        return cls._instance

    def mark_dirty(self) -> None:
        self._dirty = True

    def rebuild(self, chunk_id_text_pairs: List[tuple[str, str]]) -> None:
        with _lock:
            if not chunk_id_text_pairs:
                self._bm25 = None
                self._chunk_ids = []
                self._dirty = False
                return

            corpus_tokens = [_tokenize(text) for _, text in chunk_id_text_pairs]
            self._chunk_ids = [cid for cid, _ in chunk_id_text_pairs]
            self._bm25 = BM25Okapi(corpus_tokens)
            self._dirty = False

            logger.info("Rebuilt BM25 index over %d chunks", len(self._chunk_ids))

    @property
    def is_dirty(self) -> bool:
        return self._dirty

    def search(self, query: str, top_k: int) -> List[tuple[str, float]]:
        if self._bm25 is None or not self._chunk_ids:
            return []

        scores = self._bm25.get_scores(_tokenize(query))
        top_indices = scores.argsort()[::-1][:top_k]

        return [
            (self._chunk_ids[i], float(scores[i]))
            for i in top_indices
            if scores[i] > 0
        ]


def get_bm25_store() -> BM25Store:
    return BM25Store()