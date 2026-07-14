"""Persistent FAISS vector store."""
from __future__ import annotations

import pickle
import threading
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np

from backend.config import settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)

_lock = threading.Lock()


class FaissVectorStore:
    """Singleton persistent FAISS index with chunk ID mapping."""

    _instance: Optional["FaissVectorStore"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with _lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, dim: int | None = None) -> None:
        if self._initialized:
            return

        self.dim = dim or settings.embedding_dim
        self.index_path = Path(settings.faiss_dir) / "index.faiss"
        self.meta_path = Path(settings.faiss_dir) / "chunk_ids.pkl"
        self.index: faiss.Index
        self.chunk_ids: List[str] = []

        self._load_or_create()
        self._initialized = True

    def _load_or_create(self) -> None:
        with _lock:
            if self.index_path.exists() and self.meta_path.exists():
                logger.info("Loading existing FAISS index from %s", self.index_path)
                self.index = faiss.read_index(str(self.index_path))

                with open(self.meta_path, "rb") as f:
                    self.chunk_ids = pickle.load(f)
            else:
                logger.info("Creating new FAISS index (dim=%d)", self.dim)
                base_index = faiss.IndexFlatIP(self.dim)
                self.index = faiss.IndexIDMap2(base_index)
                self.chunk_ids = []

    def _persist(self) -> None:
        faiss.write_index(self.index, str(self.index_path))

        with open(self.meta_path, "wb") as f:
            pickle.dump(self.chunk_ids, f)

    def add(self, vectors: np.ndarray, chunk_ids: List[str]) -> List[int]:
        """Add vectors to the index and return their assigned IDs."""
        if vectors.shape[0] == 0:
            return []

        with _lock:
            start_id = len(self.chunk_ids)
            ids = np.arange(start_id, start_id + vectors.shape[0], dtype="int64")

            self.index.add_with_ids(vectors.astype("float32"), ids)
            self.chunk_ids.extend(chunk_ids)

            self._persist()

        logger.info(
            "Added %d vectors to FAISS index (total=%d)",
            vectors.shape[0],
            len(self.chunk_ids),
        )

        return ids.tolist()

    def search(self, query_vector: np.ndarray, top_k: int) -> List[tuple[str, float]]:
        """Return the most similar chunks for a query vector."""
        if not self.chunk_ids:
            return []

        query = query_vector.reshape(1, -1).astype("float32")
        scores, ids = self.index.search(query, min(top_k, len(self.chunk_ids)))

        results: List[tuple[str, float]] = []

        for score, row_id in zip(scores[0], ids[0]):
            if row_id == -1:
                continue
            results.append((self.chunk_ids[row_id], float(score)))

        return results

    @property
    def total_vectors(self) -> int:
        return len(self.chunk_ids)

    def reset(self) -> None:
        """Clear the FAISS index."""
        with _lock:
            base_index = faiss.IndexFlatIP(self.dim)
            self.index = faiss.IndexIDMap2(base_index)
            self.chunk_ids = []
            self._persist()


def get_vector_store() -> FaissVectorStore:
    return FaissVectorStore()