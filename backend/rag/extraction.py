"""Text extraction utilities for supported document formats."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Tuple

import docx
import fitz

from backend.utils.logger import get_logger

logger = get_logger(__name__)


class UnsupportedFileTypeError(Exception):
    pass


class TextExtractor:
    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

    def extract(self, file_path: str) -> List[Tuple[int, str]]:
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix not in self.SUPPORTED_EXTENSIONS:
            raise UnsupportedFileTypeError(f"Unsupported file type: {suffix}")

        if suffix == ".pdf":
            return self._extract_pdf(path)
        if suffix == ".docx":
            return self._extract_docx(path)
        return self._extract_txt(path)

    @staticmethod
    def _extract_pdf(path: Path) -> List[Tuple[int, str]]:
        pages: List[Tuple[int, str]] = []

        with fitz.open(path) as pdf:
            for i, page in enumerate(pdf, start=1):
                text = page.get_text("text")
                if text.strip():
                    pages.append((i, text))

        logger.info("Extracted %d pages from PDF %s", len(pages), path.name)
        return pages

    @staticmethod
    def _extract_docx(path: Path) -> List[Tuple[int, str]]:
        document = docx.Document(str(path))
        full_text = "\n".join(
            p.text for p in document.paragraphs if p.text.strip()
        )

        logger.info(
            "Extracted %d characters from DOCX %s",
            len(full_text),
            path.name,
        )

        return [(1, full_text)]

    @staticmethod
    def _extract_txt(path: Path) -> List[Tuple[int, str]]:
        text = path.read_text(encoding="utf-8", errors="ignore")

        logger.info(
            "Extracted %d characters from TXT %s",
            len(text),
            path.name,
        )

        return [(1, text)]

    @staticmethod
    def compute_content_hash(pages: List[Tuple[int, str]]) -> str:
        joined = "".join(text for _, text in pages)
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()

    @staticmethod
    def total_char_count(pages: List[Tuple[int, str]]) -> int:
        return sum(len(text) for _, text in pages)