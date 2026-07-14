"""Prompt builder for LLM verification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class RetrievedCandidate:
    text: str
    source_document_name: str
    page_number: int
    similarity_score: float


SYSTEM_PROMPT = """You are a plagiarism-detection classifier used inside an \
automated pipeline. You will be given one QUERY SENTENCE from a document \
being scanned, and one or more CANDIDATE passages retrieved from a reference \
corpus because they are semantically similar.

Decide how the QUERY SENTENCE relates to the single most relevant CANDIDATE:

- "Exact Copy": the query sentence is verbatim or almost verbatim (only \
trivial punctuation/casing differences) to a candidate.
- "Near Copy": the query sentence shares most of its wording and structure \
with a candidate, with minor word substitutions or reordering.
- "Paraphrased": the query sentence expresses the same idea/meaning as a \
candidate but with substantially different wording and/or structure.
- "Original": no candidate expresses the same idea; the sentence is not \
plagiarized.

Respond with ONLY a single JSON object, no prose, no markdown fences, no \
explanation outside the JSON. The JSON schema is exactly:

{
  "classification": "Exact Copy" | "Near Copy" | "Paraphrased" | "Original",
  "confidence_score": <float between 0 and 1>,
  "reason": "<one concise sentence explaining the decision>",
  "matched_text": "<the exact candidate text that was matched, or null if Original>"
}
"""


def build_verification_prompt(
    query_sentence: str,
    candidates: List[RetrievedCandidate],
) -> str:
    """Build the verification prompt."""
    lines = [f'QUERY SENTENCE:\n"{query_sentence}"\n', "CANDIDATES:"]

    for i, candidate in enumerate(candidates, start=1):
        lines.append(
            f'{i}. (source="{candidate.source_document_name}", '
            f'page={candidate.page_number}, '
            f'similarity={candidate.similarity_score:.3f})\n'
            f'"{candidate.text}"'
        )

    lines.append(
        "\nReturn ONLY the JSON object described in the system prompt, "
        "evaluating the QUERY SENTENCE against the closest matching candidate."
    )

    return "\n".join(lines)