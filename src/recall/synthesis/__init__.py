"""Synthesis and generation guardrails engine for Recall."""

from recall.synthesis.citations import extract_and_verify_citations
from recall.synthesis.generator import DEFAULT_SYSTEM_PROMPT, Synthesizer
from recall.synthesis.models import Citation, SynthesizedResponse
from recall.synthesis.sandbox import build_sandboxed_context, sanitize_passage_for_prompt

__all__ = [
    "Citation",
    "SynthesizedResponse",
    "DEFAULT_SYSTEM_PROMPT",
    "Synthesizer",
    "extract_and_verify_citations",
    "build_sandboxed_context",
    "sanitize_passage_for_prompt",
]
