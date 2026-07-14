"""Handles document ingestion, indexing, and corpus management."""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import List

from sqlalchemy.orm import Session

from backend.config import settings
from backend.database.models import Chunk, Document, Embedding
from backend.rag.chunking import SemanticChunker
from backend.rag.embeddings import get_embedding_service
from backend.rag.extraction import TextExtractor
from backend.vectorstore.bm25_store import get_bm25_store
from backend.vectorstore.faiss_store import get_vector_store
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentService:
    """Handles document upload, ingestion, and corpus indexing."""

    def __init__(self) -> None:
        self.extractor = TextExtractor()
        self.chunker = SemanticChunker()
        self.embedding_service = get_embedding_service()
        self.vector_store = get_vector_store()
        self.bm25_store = get_bm25_store()

    def save_upload(self, filename: str, file_bytes: bytes, doc_type: str = "scan") -> Path:
        """Save an uploaded document to storage."""
        suffix = Path(filename).suffix.lower()
        safe_name = f"{uuid.uuid4().hex}{suffix}"
        target_dir = Path(settings.corpus_dir) if doc_type == "corpus" else Path(settings.upload_dir)
        target_path = target_dir / safe_name

        with open(target_path, "wb") as f:
            f.write(file_bytes)

        logger.info("Saved upload '%s' -> %s", filename, target_path)
        return target_path

    def find_duplicate(self, db: Session, content_hash: str) -> Document | None:
        return db.query(Document).filter(Document.content_hash == content_hash).first()

    def ingest(
        self,
        db: Session,
        filename: str,
        file_bytes: bytes,
        doc_type: str = "scan",
    ) -> tuple[Document, int, bool]:
        """Save and optionally index a document."""

        saved_path = self.save_upload(filename, file_bytes, doc_type)
        suffix = saved_path.suffix.lower().lstrip(".")

        pages = self.extractor.extract(str(saved_path))
        content_hash = self.extractor.compute_content_hash(pages)
        char_count = self.extractor.total_char_count(pages)

        existing = self.find_duplicate(db, content_hash)
        if existing is not None and doc_type == "corpus":
            logger.info(
                "Duplicate corpus document detected (hash=%s); skipping re-index",
                content_hash[:12],
            )
            saved_path.unlink(missing_ok=True)
            return existing, 0, True

        document = Document(
            filename=filename,
            file_path=str(saved_path),
            file_type=suffix,
            doc_type=doc_type,
            num_pages=len(pages) or 1,
            char_count=char_count,
            content_hash=content_hash,
        )

        db.add(document)
        db.flush()

        num_indexed = 0
        if doc_type == "corpus":
            num_indexed = self._index_as_corpus(db, document, pages)

        db.commit()
        db.refresh(document)

        return document, num_indexed, False

    def _index_as_corpus(
        self,
        db: Session,
        document: Document,
        pages: List[tuple[int, str]],
    ) -> int:
        """Chunk, embed, and index a corpus document."""

        chunk_records = self.chunker.chunk_pages(pages)
        if not chunk_records:
            return 0

        texts = [c.text for c in chunk_records]
        vectors = self.embedding_service.embed(texts)

        chunk_rows: List[Chunk] = []

        for record in chunk_records:
            chunk_rows.append(
                Chunk(
                    document_id=document.id,
                    chunk_index=record.chunk_index,
                    page_number=record.page_number,
                    text=record.text,
                )
            )

        db.add_all(chunk_rows)
        db.flush()

        chunk_ids = [row.id for row in chunk_rows]
        faiss_ids = self.vector_store.add(vectors, chunk_ids)

        for row, faiss_id, vector in zip(chunk_rows, faiss_ids, vectors):
            row.faiss_vector_id = faiss_id

            db.add(
                Embedding(
                    chunk_id=row.id,
                    model_name=settings.embedding_model_name,
                    vector=vector.tobytes(),
                    dim=vector.shape[0],
                )
            )

        self.bm25_store.mark_dirty()

        logger.info(
            "Indexed %d chunks for document '%s'",
            len(chunk_rows),
            document.filename,
        )

        return len(chunk_rows)

    def refresh_bm25_if_needed(self, db: Session) -> None:
        if self.bm25_store.is_dirty:
            rows = db.query(Chunk.id, Chunk.text).all()
            self.bm25_store.rebuild([(row.id, row.text) for row in rows])

    def library_is_empty(self, db: Session) -> bool:
        """Return True if the corpus contains no indexed documents."""
        return db.query(Document).filter(Document.doc_type == "corpus").count() == 0

    def promote_to_corpus(self, db: Session, document_id: str) -> int:
        """Add an uploaded scan document to the reference corpus."""

        document = db.query(Document).filter(Document.id == document_id).first()
        if document is None:
            raise ValueError(f"Document {document_id} not found")

        existing_chunks = (
            db.query(Chunk)
            .filter(Chunk.document_id == document.id)
            .count()
        )

        if existing_chunks > 0:
            logger.info(
                "Document '%s' is already indexed in the library",
                document.filename,
            )
            document.doc_type = "corpus"
            db.commit()
            return existing_chunks

        pages = self.extractor.extract(document.file_path)
        num_indexed = self._index_as_corpus(db, document, pages)

        document.doc_type = "corpus"

        db.commit()
        db.refresh(document)

        return num_indexed


def get_document_service() -> DocumentService:
    return DocumentService()