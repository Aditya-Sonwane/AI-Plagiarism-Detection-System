"""NLTK-based text preprocessing utilities."""

from __future__ import annotations

import re
from typing import List

import nltk

from backend.utils.logger import get_logger

logger = get_logger(__name__)

_REQUIRED_NLTK_RESOURCES = [
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords", "stopwords"),
]


def ensure_nltk_resources() -> None:
    for resource_path, package_name in _REQUIRED_NLTK_RESOURCES:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            logger.info("Downloading missing NLTK resource: %s", package_name)
            try:
                nltk.download(package_name, quiet=True)
            except Exception as exc:
                logger.warning(
                    "Could not download NLTK resource %s: %s",
                    package_name,
                    exc,
                )


class TextPreprocessor:
    def __init__(self) -> None:
        ensure_nltk_resources()

    @staticmethod
    def clean_text(text: str) -> str:
        text = text.replace("\x0c", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{2,}", "\n", text)
        text = re.sub(r"-\n(\w)", r"\1", text)
        text = re.sub(r"\n", " ", text)
        return text.strip()

    def sentence_tokenize(self, text: str) -> List[str]:
        cleaned = self.clean_text(text)

        if not cleaned:
            return []

        try:
            sentences = nltk.sent_tokenize(cleaned)
        except LookupError:
            ensure_nltk_resources()
            sentences = nltk.sent_tokenize(cleaned)

        return [
            sentence.strip()
            for sentence in sentences
            if len(sentence.strip()) >= 15
        ]