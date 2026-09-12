"""Context sandboxing and prompt injection defense for untrusted document passages."""

from __future__ import annotations

import re
from recall.core.models import SearchResult

# Markers commonly used in prompt injection jailbreaks
_INJECTION_PATTERNS = [
    r"(?i)\bignore\s+(?:all\s+)?(?:previous|prior)\s+instructions\b",
    r"(?i)\byou\s+are\s+now\s+(?:in\s+developer\s+mode|dan\s+mode|unrestricted)\b",
    r"(?i)\bdisregard\s+(?:all\s+)?(?:rules|system\s+instructions)\b",
    r"<\|im_start\|>",
    r"<\|im_end\|>",
    r"\[/?INST\]",
    r"\[/?SYS\]",
]


def sanitize_passage_for_prompt(text: str) -> str:
    """Sanitizes text extracted from third-party documents to neutralize
    XML breakout attacks and indirect prompt injection attempts.
    """
    sanitized = text

    # Escape rogue context XML closing/opening tags
    sanitized = sanitized.replace("</context_document>", "&lt;/context_document&gt;")
    sanitized = sanitized.replace("<context_document", "&lt;context_document")
    sanitized = sanitized.replace("</context_documents>", "&lt;/context_documents&gt;")
    sanitized = sanitized.replace("<context_documents>", "&lt;context_documents&gt;")

    # Defang dangerous injection instructions
    for pattern in _INJECTION_PATTERNS:
        sanitized = re.sub(
            pattern,
            "[BLOCKED_INJECTION_ATTEMPT]",
            sanitized,
        )

    return sanitized


def build_sandboxed_context(
    candidates: list[SearchResult],
) -> tuple[str, dict[int, SearchResult]]:
    """Encapsulates retrieved search candidates into structured XML elements with provenance attributes.

    Returns:
        tuple of:
          - sandboxed_xml_string: Formatted XML string ready for insertion into the LLM prompt.
          - candidate_map: Mapping from 1-indexed document numbers to the corresponding SearchResult.
    """
    if not candidates:
        return "<context_documents>\n(No relevant documents found.)\n</context_documents>", {}

    candidate_map: dict[int, SearchResult] = {}
    doc_blocks: list[str] = ["<context_documents>"]

    for idx, candidate in enumerate(candidates, start=1):
        candidate_map[idx] = candidate

        meta = candidate.metadata
        source_attr = meta.source_uri or f"doc_{meta.doc_id}"
        page_attr = f' page="{meta.page_number}"' if meta.page_number is not None else ""

        sanitized_text = sanitize_passage_for_prompt(candidate.text)

        block = (
            f'<context_document id="{idx}" source="{source_attr}"{page_attr}>\n'
            f"{sanitized_text}\n"
            f"</context_document>"
        )
        doc_blocks.append(block)

    doc_blocks.append("</context_documents>")
    return "\n".join(doc_blocks), candidate_map
