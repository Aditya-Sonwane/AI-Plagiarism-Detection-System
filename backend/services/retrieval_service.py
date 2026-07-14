"""Hybrid retrieval service combining semantic and lexical search."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from sqlalchemy.orm import Session

from backend.config import settings
from backend.database.models import Chunk, Document
from backend.rag.embeddings import get_embedding_service
from backend.rag.reranker import get_reranker
from backend.vectorstore.bm25_store import get_bm25_store
from backend.vectorstore.faiss_store import get_vector_store
from backend.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    document_id: str
    document_name: str
    page_number: int
    similarity_score: float


class RetrievalService:
    """Retrieves the most relevant chunks for a query sentence."""

    def __init__(self) -> None:
        self.embedding_service = get_embedding_service()
        self.vector_store = get_vector_store()
        self.bm25_store = get_bm25_store()
        self.reranker = get_reranker()

    def retrieve(
        self,
        db: Session,
        query_sentence: str,
        top_k: int | None = None,
        exclude_document_id: str | None = None,
    ) -> List[RetrievedChunk]:
        top_k = top_k or settings.top_k
        fetch_k = max(top_k * 4, 20)

        query_vector = self.embedding_service.embed_one(query_sentence)
        faiss_hits = self.vector_store.search(query_vector, fetch_k)

        bm25_hits: List[tuple[str, float]] = []
        if settings.use_hybrid_retrieval:
            bm25_hits = self.bm25_store.search(query_sentence, fetch_k)

        fused_scores: Dict[str, float] = {}

        for rank, (chunk_id, _) in enumerate(faiss_hits):
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (60 + rank)

        for rank, (chunk_id, _) in enumerate(bm25_hits):
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (60 + rank)

        if not fused_scores:
            return []

        raw_similarity = {cid: score for cid, score in faiss_hits}

        candidate_ids = sorted(
            fused_scores.keys(),
            key=lambda cid: fused_scores[cid],
            reverse=True,
        )[:fetch_k]

        rows = (
            db.query(Chunk, Document)
            .join(Document, Chunk.document_id == Document.id)
            .filter(Chunk.id.in_(candidate_ids))
            .all()
        )

        if exclude_document_id:
            rows = [(chunk, doc) for chunk, doc in rows if doc.id != exclude_document_id]

        chunk_lookup = {chunk.id: (chunk, doc) for chunk, doc in rows}

        ordered_ids = [cid for cid in candidate_ids if cid in chunk_lookup]

        if settings.use_cross_encoder_rerank and ordered_ids:
            pairs = [(cid, chunk_lookup[cid][0].text) for cid in ordered_ids]
            reranked = self.reranker.rerank(query_sentence, pairs, top_k)

            results: List[RetrievedChunk] = []

            for chunk_id, text, rerank_score in reranked:
                chunk, document = chunk_lookup[chunk_id]

                similarity = raw_similarity.get(chunk_id)

                if similarity is None:
                    import math

                    similarity = 1 / (1 + math.exp(-rerank_score))

                results.append(
                    RetrievedChunk(
                        chunk_id=chunk.id,
                        text=chunk.text,
                        document_id=document.id,
                        document_name=document.filename,
                        page_number=chunk.page_number,
                        similarity_score=similarity,
                    )
                )

            return results

        results: List[RetrievedChunk] = []

        for cid in ordered_ids[:top_k]:
            chunk, document = chunk_lookup[cid]

            results.append(
                RetrievedChunk(
                    chunk_id=chunk.id,
                    text=chunk.text,
                    document_id=document.id,
                    document_name=document.filename,
                    page_number=chunk.page_number,
                    similarity_score=raw_similarity.get(cid, fused_scores[cid]),
                )
            )

        return results


def get_retrieval_service() -> RetrievalService:
    return RetrievalService()