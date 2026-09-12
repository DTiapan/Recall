"""Data models for synthesis, citation attribution, and guardrail responses."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """Represents a validated citation linking a factual statement back to source documents."""

    doc_index: int = Field(..., description="1-indexed reference number in the context prompt")
    chunk_id: str = Field(..., description="Unique ID of the chunk providing the evidence")
    source_uri: str | None = Field(default=None, description="Path or URL to the origin document")
    page_number: int | None = Field(default=None, description="Page number if applicable")
    snippet: str = Field(..., description="Brief snippet of the supporting passage text")


class SynthesizedResponse(BaseModel):
    """Complete synthesized answer accompanied by citation audit trail and metrics."""

    answer: str = Field(..., description="Generated answer text")
    citations: list[Citation] = Field(default_factory=list, description="Verified inline citations")
    unverified_citations: list[int] = Field(
        default_factory=list,
        description="Citation indices found in text that do not correspond to provided documents",
    )
    model_name: str = Field(..., description="LLM model identifier used for synthesis")
    latency_seconds: float = Field(default=0.0, description="End-to-end generation latency in seconds")
