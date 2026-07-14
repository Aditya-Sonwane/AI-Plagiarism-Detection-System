"""API routes."""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.config import settings
from backend.database.db import get_db
from backend.database.models import Document, ScanHistory
from backend.rag.extraction import UnsupportedFileTypeError
from backend.schemas.schemas import (
    DocumentOut,
    HealthOut,
    ScanHistoryOut,
    ScanRequest,
    ScanResultOut,
    UploadResponse,
)
from backend.services.document_service import DocumentService, get_document_service
from backend.services.report_service import ReportService, get_report_service
from backend.services.scan_service import ScanService, get_scan_service
from backend.utils.logger import get_logger
from backend.vectorstore.faiss_store import get_vector_store

logger = get_logger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthOut, tags=["System"])
def health_check() -> HealthOut:
    store = get_vector_store()
    return HealthOut(
        status="ok",
        embedding_model=settings.embedding_model_name,
        llm_provider=settings.llm_provider,
        faiss_vectors_indexed=store.total_vectors,
    )


@router.post("/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_document(
    file: UploadFile = File(...),
    doc_type: str = Form("corpus"),
    db: Session = Depends(get_db),
    document_service: DocumentService = Depends(get_document_service),
) -> UploadResponse:
    if doc_type not in ("corpus", "scan"):
        raise HTTPException(400, "doc_type must be 'corpus' or 'scan'")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(400, "Uploaded file is empty")

    try:
        document, num_chunks, is_duplicate = document_service.ingest(
            db,
            file.filename,
            file_bytes,
            doc_type=doc_type,
        )
    except UnsupportedFileTypeError as exc:
        raise HTTPException(400, str(exc)) from exc

    return UploadResponse(
        document=DocumentOut.model_validate(document),
        num_chunks_indexed=num_chunks,
        is_duplicate=is_duplicate,
    )


def _build_scan_response(scan) -> ScanResultOut:
    result = ScanResultOut.model_validate(scan)

    if scan.library_was_empty:
        result.warning = (
            "Your reference library was empty when this document was scanned, so there was "
            "nothing to compare it against. A low or 0% result does not confirm the document "
            "is original. Add reference documents to the library and scan again."
        )

    return result


@router.post("/scan", response_model=ScanResultOut, tags=["Scan"])
def scan_document(
    request: ScanRequest,
    db: Session = Depends(get_db),
    scan_service: ScanService = Depends(get_scan_service),
    document_service: DocumentService = Depends(get_document_service),
) -> ScanResultOut:
    try:
        scan = scan_service.run_scan(
            db,
            document_id=request.document_id,
            top_k=request.top_k,
            llm_provider=request.llm_provider,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    if request.add_to_library:
        document_service.promote_to_corpus(db, request.document_id)

    return _build_scan_response(scan)


@router.get("/report/{scan_id}", tags=["Scan"])
def get_report(
    scan_id: str,
    db: Session = Depends(get_db),
    report_service: ReportService = Depends(get_report_service),
) -> FileResponse:
    scan = db.query(ScanHistory).filter(ScanHistory.id == scan_id).first()

    if scan is None:
        raise HTTPException(404, f"Scan {scan_id} not found")

    report_path = scan.report_path
    if not report_path:
        report_path = report_service.generate(db, scan)

    return FileResponse(
        report_path,
        media_type="application/pdf",
        filename=f"plagiarism_report_{scan_id}.pdf",
    )


@router.get("/scan/{scan_id}", response_model=ScanResultOut, tags=["Scan"])
def get_scan_result(
    scan_id: str,
    db: Session = Depends(get_db),
) -> ScanResultOut:
    scan = db.query(ScanHistory).filter(ScanHistory.id == scan_id).first()

    if scan is None:
        raise HTTPException(404, f"Scan {scan_id} not found")

    return _build_scan_response(scan)


@router.get("/history", response_model=List[ScanHistoryOut], tags=["Scan"])
def get_history(
    limit: int = 50,
    db: Session = Depends(get_db),
) -> List[ScanHistoryOut]:
    scans = (
        db.query(ScanHistory)
        .order_by(ScanHistory.created_at.desc())
        .limit(limit)
        .all()
    )

    return [ScanHistoryOut.model_validate(scan) for scan in scans]


@router.get("/documents", response_model=List[DocumentOut], tags=["Documents"])
def list_documents(
    doc_type: str | None = None,
    db: Session = Depends(get_db),
) -> List[DocumentOut]:
    query = db.query(Document)

    if doc_type:
        query = query.filter(Document.doc_type == doc_type)

    documents = query.order_by(Document.created_at.desc()).all()

    return [DocumentOut.model_validate(document) for document in documents]


@router.delete("/documents/{document_id}", tags=["Documents"])
def delete_document(
    document_id: str,
    db: Session = Depends(get_db),
) -> dict:
    document = db.query(Document).filter(Document.id == document_id).first()

    if document is None:
        raise HTTPException(404, f"Document {document_id} not found")

    db.delete(document)
    db.commit()

    return {"deleted": document_id}