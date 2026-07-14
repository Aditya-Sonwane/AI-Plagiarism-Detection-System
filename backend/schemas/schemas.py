"""Pydantic schemas for the API."""

from __future__ import annotations

import datetime as dt
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Classification(str, Enum):
    EXACT_COPY = "Exact Copy"
    NEAR_COPY = "Near Copy"
    PARAPHRASED = "Paraphrased"
    ORIGINAL = "Original"


class DocumentOut(BaseModel):
    id: str
    filename: str
    file_type: str
    doc_type: str
    num_pages: int
    char_count: int
    created_at: dt.datetime

    class Config:
        from_attributes = True


class UploadResponse(BaseModel):
    document: DocumentOut
    num_chunks_indexed: int
    is_duplicate: bool = False


class LLMVerificationResult(BaseModel):
    classification: Classification
    confidence_score: float = Field(ge=0.0, le=1.0)
    reason: str
    matched_text: Optional[str] = None


class MatchOut(BaseModel):
    query_sentence: str
    query_sentence_index: int
    matched_text: Optional[str] = None
    source_document_id: Optional[str] = None
    source_document_name: Optional[str] = None
    source_page_number: Optional[int] = None
    similarity_score: float
    classification: Classification
    confidence_score: float
    reason: Optional[str] = None

    class Config:
        from_attributes = True


class ScanRequest(BaseModel):
    document_id: str = Field(..., description="Document ID to scan")
    top_k: Optional[int] = None
    llm_provider: Optional[str] = None
    add_to_library: bool = Field(
        default=False,
        description="Add the document to the reference library after scanning.",
    )


class ScanResultOut(BaseModel):
    id: str
    document_id: str
    overall_plagiarism_pct: float
    exact_copy_pct: float
    near_copy_pct: float
    paraphrased_pct: float
    original_pct: float
    total_sentences: int
    execution_time_seconds: float
    embedding_model: str
    warning: Optional[str] = Field(
        default=None,
        description="Warning message for special scan conditions.",
    )
    llm_provider: str
    created_at: dt.datetime
    matches: list[MatchOut] = []

    class Config:
        from_attributes = True


class ScanHistoryOut(BaseModel):
    id: str
    document_id: str
    overall_plagiarism_pct: float
    total_sentences: int
    execution_time_seconds: float
    created_at: dt.datetime

    class Config:
        from_attributes = True


class HealthOut(BaseModel):
    status: str
    embedding_model: str
    llm_provider: str
    faiss_vectors_indexed: int