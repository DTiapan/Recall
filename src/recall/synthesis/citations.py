"""Citation extraction, parsing, and grounding verification."""

from __future__ import annotations

import re
from recall.core.models import SearchResult
from recall.synthesis.models import Citation

# Regex matching [Doc 1], [Doc 1, p. 5], [Doc 2, p.12], [Doc 3, page 4]
CITATION_REGEX = re.compile(
    r"\[Doc\s*(\d+)(?:,\s*(?:p\.?|page)\s*(\d+))?\]",
    re.IGNORECASE,
)


def extract_and_verify_citations(
    answer: str,
    candidate_map: dict[int, SearchResult],
) -> tuple[list[Citation], list[int]]:
    """Extracts inline citation references from the generated answer and verifies them
    against the provided context documents.

    Args:
        answer: Generated text response from the LLM containing [Doc X] tags.
        candidate_map: Mapping from document index (1, 2, ...) to the SearchResult in context.

    Returns:
        tuple of:
          - verified_citations: List of Citation objects successfully linked to candidate documents.
          - unverified_citations: List of integer doc indices cited in text that were not in candidate_map.
    """
    matches = CITATION_REGEX.findall(answer)
    if not matches:
        return [], []

    seen_indices: set[tuple[int, int | None]] = set()
    verified: list[Citation] = []
    unverified: list[int] = []

    for doc_idx_str, page_str in matches:
        doc_idx = int(doc_idx_str)
        page_num = int(page_str) if page_str else None

        key = (doc_idx, page_num)
        if key in seen_indices:
            continue
        seen_indices.add(key)

        if doc_idx in candidate_map:
            candidate = candidate_map[doc_idx]
            resolved_page = page_num if page_num is not None else candidate.metadata.page_number
            snippet = candidate.text[:200].replace("\n", " ").strip()

            citation = Citation(
                doc_index=doc_idx,
                chunk_id=candidate.chunk_id,
                source_uri=candidate.metadata.source_uri,
                page_number=resolved_page,
                snippet=snippet,
            )
            verified.append(citation)
        else:
            unverified.append(doc_idx)

    return verified, unverified
