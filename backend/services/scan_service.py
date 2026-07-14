"""Runs the end-to-end plagiarism scanning pipeline."""
from __future__ import annotations

import time
from typing import List

from sqlalchemy.orm import Session

from backend.config import settings
from backend.database.models import Document, Match, ScanHistory
from backend.llm.llm_client import get_llm_client
from backend.llm.prompt_builder import RetrievedCandidate
from backend.rag.chunking import SemanticChunker
from backend.rag.extraction import TextExtractor
from backend.schemas.schemas import Classification
from backend.services.document_service import get_document_service
from backend.services.retrieval_service import get_retrieval_service
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class ScanService:
    """Runs the complete plagiarism detection pipeline."""

    def __init__(self) -> None:
        self.extractor = TextExtractor()
        self.chunker = SemanticChunker()
        self.retrieval_service = get_retrieval_service()
        self.document_service = get_document_service()

    def run_scan(
        self,
        db: Session,
        document_id: str,
        top_k: int | None = None,
        llm_provider: str | None = None,
    ) -> ScanHistory:
        start_time = time.perf_counter()
        top_k = top_k or settings.top_k

        document = db.query(Document).filter(Document.id == document_id).first()
        if document is None:
            raise ValueError(f"Document {document_id} not found")

        self.document_service.refresh_bm25_if_needed(db)
        library_was_empty = self.document_service.library_is_empty(db)

        pages = self.extractor.extract(document.file_path)
        sentence_stream = self.chunker.sentence_stream(pages)

        llm_client = get_llm_client(llm_provider)

        match_rows: List[Match] = []
        counts = {
            Classification.EXACT_COPY: 0,
            Classification.NEAR_COPY: 0,
            Classification.PARAPHRASED: 0,
            Classification.ORIGINAL: 0,
        }

        for idx, (page_number, sentence) in enumerate(sentence_stream):
            retrieved = self.retrieval_service.retrieve(
                db,
                sentence,
                top_k=top_k,
                exclude_document_id=document.id,
            )

            candidates = [
                RetrievedCandidate(
                    text=r.text,
                    source_document_name=r.document_name,
                    page_number=r.page_number,
                    similarity_score=r.similarity_score,
                )
                for r in retrieved
            ]

            top_similarity = retrieved[0].similarity_score if retrieved else 0.0

            if not retrieved or top_similarity < settings.similarity_threshold:
                classification = Classification.ORIGINAL
                confidence = 1.0 - top_similarity if retrieved else 1.0
                reason = "No sufficiently similar source material was found."
                matched_text = None
                source_doc_id = None
                source_doc_name = None
                source_page = None
            else:
                result = llm_client.verify(sentence, candidates, top_similarity)
                classification = result.classification
                confidence = result.confidence_score
                reason = result.reason
                matched_text = result.matched_text
                source_doc_id = retrieved[0].document_id
                source_doc_name = retrieved[0].document_name
                source_page = retrieved[0].page_number

            counts[classification] += 1

            match_rows.append(
                Match(
                    scan_id=None,
                    query_sentence=sentence,
                    query_sentence_index=idx,
                    matched_text=matched_text,
                    source_document_id=source_doc_id,
                    source_document_name=source_doc_name,
                    source_page_number=source_page,
                    similarity_score=top_similarity,
                    classification=classification.value,
                    confidence_score=confidence,
                    reason=reason,
                )
            )

        total = len(sentence_stream) or 1

        exact_pct = 100.0 * counts[Classification.EXACT_COPY] / total
        near_pct = 100.0 * counts[Classification.NEAR_COPY] / total
        para_pct = 100.0 * counts[Classification.PARAPHRASED] / total
        original_pct = 100.0 * counts[Classification.ORIGINAL] / total
        overall_pct = exact_pct + near_pct + para_pct

        elapsed = time.perf_counter() - start_time

        scan = ScanHistory(
            document_id=document.id,
            overall_plagiarism_pct=round(overall_pct, 2),
            exact_copy_pct=round(exact_pct, 2),
            near_copy_pct=round(near_pct, 2),
            paraphrased_pct=round(para_pct, 2),
            original_pct=round(original_pct, 2),
            total_sentences=len(sentence_stream),
            execution_time_seconds=round(elapsed, 3),
            embedding_model=settings.embedding_model_name,
            llm_provider=llm_provider or settings.llm_provider,
            library_was_empty=library_was_empty,
        )

        db.add(scan)
        db.flush()

        for row in match_rows:
            row.scan_id = scan.id

        db.add_all(match_rows)
        db.commit()
        db.refresh(scan)

        logger.info(
            "Scan complete for document '%s': %.2f%% plagiarism over %d sentences in %.2fs",
            document.filename,
            overall_pct,
            len(sentence_stream),
            elapsed,
        )

        return scan


def get_scan_service() -> ScanService:
    return ScanService()