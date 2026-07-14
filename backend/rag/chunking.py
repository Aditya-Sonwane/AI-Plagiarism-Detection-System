"""Sentence-window chunking for document indexing and scanning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from backend.config import settings
from backend.rag.preprocessing import TextPreprocessor
from backend.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ChunkRecord:
    chunk_index: int
    page_number: int
    text: str
    sentences: List[str] = field(default_factory=list)


class SemanticChunker:
    def __init__(
        self,
        window_size: int | None = None,
        overlap: int | None = None,
    ) -> None:
        self.window_size = window_size or settings.chunk_size_sentences
        self.overlap = overlap if overlap is not None else settings.chunk_overlap_sentences
        self.preprocessor = TextPreprocessor()

        if self.overlap >= self.window_size:
            raise ValueError("chunk_overlap_sentences must be smaller than chunk_size_sentences")

    def chunk_pages(self, pages: List[tuple[int, str]]) -> List[ChunkRecord]:
        chunks: List[ChunkRecord] = []
        chunk_idx = 0
        step = self.window_size - self.overlap

        for page_number, raw_text in pages:
            sentences = self.preprocessor.sentence_tokenize(raw_text)
            if not sentences:
                continue

            i = 0
            while i < len(sentences):
                window = sentences[i : i + self.window_size]
                if not window:
                    break

                chunks.append(
                    ChunkRecord(
                        chunk_index=chunk_idx,
                        page_number=page_number,
                        text=" ".join(window),
                        sentences=window,
                    )
                )

                chunk_idx += 1

                if i + self.window_size >= len(sentences):
                    break

                i += step

        logger.info("Built %d chunks from %d pages", len(chunks), len(pages))
        return chunks

    def sentence_stream(self, pages: List[tuple[int, str]]) -> List[tuple[int, str]]:
        stream: List[tuple[int, str]] = []

        for page_number, raw_text in pages:
            for sentence in self.preprocessor.sentence_tokenize(raw_text):
                stream.append((page_number, sentence))

        return stream