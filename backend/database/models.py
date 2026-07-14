"""Database models."""

from __future__ import annotations

import datetime as dt
import uuid

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from backend.database.db import Base


def _uuid() -> str:
    return str(uuid.uuid4())


class Document(Base):
    """Uploaded document."""

    __tablename__ = "documents"

    id = Column(String, primary_key=True, default=_uuid)
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    doc_type = Column(String, nullable=False, default="scan")
    num_pages = Column(Integer, default=1)
    char_count = Column(Integer, default=0)
    content_hash = Column(String, index=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow)

    chunks = relationship(
        "Chunk",
        back_populates="document",
        cascade="all, delete-orphan",
    )


class Chunk(Base):
    """Semantic text chunk."""

    __tablename__ = "chunks"

    id = Column(String, primary_key=True, default=_uuid)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    page_number = Column(Integer, default=1)
    text = Column(Text, nullable=False)
    faiss_vector_id = Column(Integer, nullable=True, index=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow)

    document = relationship("Document", back_populates="chunks")
    embedding = relationship(
        "Embedding",
        back_populates="chunk",
        uselist=False,
        cascade="all, delete-orphan",
    )


class Embedding(Base):
    """Embedding vector."""

    __tablename__ = "embeddings"

    id = Column(String, primary_key=True, default=_uuid)
    chunk_id = Column(String, ForeignKey("chunks.id"), nullable=False, unique=True)
    model_name = Column(String, nullable=False)
    vector = Column(LargeBinary, nullable=False)
    dim = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=dt.datetime.utcnow)

    chunk = relationship("Chunk", back_populates="embedding")


class ScanHistory(Base):
    """Plagiarism scan result."""

    __tablename__ = "scan_history"

    id = Column(String, primary_key=True, default=_uuid)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    overall_plagiarism_pct = Column(Float, default=0.0)
    exact_copy_pct = Column(Float, default=0.0)
    near_copy_pct = Column(Float, default=0.0)
    paraphrased_pct = Column(Float, default=0.0)
    original_pct = Column(Float, default=0.0)
    total_sentences = Column(Integer, default=0)
    execution_time_seconds = Column(Float, default=0.0)
    embedding_model = Column(String)
    llm_provider = Column(String)
    report_path = Column(String, nullable=True)
    library_was_empty = Column(Boolean, default=False)
    created_at = Column(DateTime, default=dt.datetime.utcnow)

    matches = relationship(
        "Match",
        back_populates="scan",
        cascade="all, delete-orphan",
    )


class Match(Base):
    """Sentence-level plagiarism match."""

    __tablename__ = "matches"

    id = Column(String, primary_key=True, default=_uuid)
    scan_id = Column(String, ForeignKey("scan_history.id"), nullable=False)
    query_sentence = Column(Text, nullable=False)
    query_sentence_index = Column(Integer, default=0)
    matched_text = Column(Text, nullable=True)
    source_document_id = Column(String, nullable=True)
    source_document_name = Column(String, nullable=True)
    source_page_number = Column(Integer, nullable=True)
    similarity_score = Column(Float, default=0.0)
    classification = Column(String, default="Original")
    confidence_score = Column(Float, default=0.0)
    reason = Column(Text, nullable=True)

    scan = relationship("ScanHistory", back_populates="matches")