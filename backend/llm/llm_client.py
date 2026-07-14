"""LLM clients for plagiarism verification."""
from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from typing import List, Optional

import requests

from backend.config import settings
from backend.llm.prompt_builder import RetrievedCandidate, SYSTEM_PROMPT, build_verification_prompt
from backend.schemas.schemas import Classification, LLMVerificationResult
from backend.utils.logger import get_logger

logger = get_logger(__name__)


def _extract_json(raw: str) -> dict:
    """Extract a JSON object from an LLM response."""
    raw = raw.strip()
    raw = re.sub(r"^```(json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in LLM response: {raw[:200]}")
    return json.loads(match.group(0))


class BaseLLMClient(ABC):
    @abstractmethod
    def _call(self, system_prompt: str, user_prompt: str) -> str:
        """Execute an LLM request."""

    def verify(
        self,
        query_sentence: str,
        candidates: List[RetrievedCandidate],
        top_similarity: float,
    ) -> LLMVerificationResult:
        if not candidates:
            return LLMVerificationResult(
                classification=Classification.ORIGINAL,
                confidence_score=1.0,
                reason="No semantically similar candidates were found in the corpus.",
                matched_text=None,
            )

        user_prompt = build_verification_prompt(query_sentence, candidates)

        last_error: Optional[Exception] = None
        for attempt in range(1, settings.llm_max_retries + 2):
            try:
                raw = self._call(SYSTEM_PROMPT, user_prompt)
                data = _extract_json(raw)
                return LLMVerificationResult(
                    classification=Classification(data["classification"]),
                    confidence_score=float(data.get("confidence_score", 0.5)),
                    reason=str(data.get("reason", "")),
                    matched_text=data.get("matched_text"),
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning("LLM verification attempt %d failed: %s", attempt, exc)
                time.sleep(0.5 * attempt)

        logger.error("LLM verification failed after retries, falling back to threshold classifier: %s", last_error)
        return ThresholdFallbackClient().verify(query_sentence, candidates, top_similarity)


class OllamaClient(BaseLLMClient):
    """Ollama client."""

    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        self.model = model or settings.ollama_model
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")

    def _call(self, system_prompt: str, user_prompt: str) -> str:
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "format": "json",
                "options": {"temperature": settings.llm_temperature},
            },
            timeout=settings.llm_timeout_seconds,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]


class OpenAIClient(BaseLLMClient):
    """OpenAI client."""

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        self.model = model or settings.openai_model
        self.api_key = api_key or settings.openai_api_key
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not configured (set PLAG_OPENAI_API_KEY).")

    def _call(self, system_prompt: str, user_prompt: str) -> str:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": settings.llm_temperature,
                "response_format": {"type": "json_object"},
            },
            timeout=settings.llm_timeout_seconds,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


class ThresholdFallbackClient(BaseLLMClient):
    """Threshold-based fallback classifier."""

    def _call(self, system_prompt: str, user_prompt: str) -> str:  # pragma: no cover
        raise NotImplementedError("ThresholdFallbackClient does not call an external LLM")

    def verify(
        self,
        query_sentence: str,
        candidates: List[RetrievedCandidate],
        top_similarity: float,
    ) -> LLMVerificationResult:
        if not candidates:
            return LLMVerificationResult(
                classification=Classification.ORIGINAL,
                confidence_score=1.0,
                reason="No semantically similar candidates were found in the corpus.",
                matched_text=None,
            )

        best = candidates[0]
        score = top_similarity

        if score >= settings.exact_copy_threshold:
            classification = Classification.EXACT_COPY
        elif score >= settings.near_copy_threshold:
            classification = Classification.NEAR_COPY
        elif score >= settings.paraphrase_threshold:
            classification = Classification.PARAPHRASED
        else:
            classification = Classification.ORIGINAL

        return LLMVerificationResult(
            classification=classification,
            confidence_score=round(min(score, 0.99), 3),
            reason=(
                f"Threshold-based classification (no LLM): cosine similarity {score:.3f} "
                f"against best-matching chunk from '{best.source_document_name}'."
            ),
            matched_text=best.text if classification != Classification.ORIGINAL else None,
        )


def get_llm_client(provider: str | None = None) -> BaseLLMClient:
    """Return the configured LLM client."""
    provider = (provider or settings.llm_provider).lower()
    if provider == "ollama":
        return OllamaClient()
    elif provider == "openai":
        return OpenAIClient()
    elif provider == "none":
        return ThresholdFallbackClient()
    raise ValueError(f"Unknown LLM provider: {provider}")
