"""Core data models for Document and Chunk representations."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any
import xxhash
from pydantic import BaseModel, Field


def compute_content_hash(text: str) -> str:
    """Computes a fast 64-bit xxhash hexadecimal digest of normalized text."""
    return xxhash.xxh64(text.strip().encode("utf-8")).hexdigest()


class Document(BaseModel):
    """Represents an ingested document before chunking."""

    id: str = Field(default_factory=lambda: hashlib.sha256(str(datetime.now(timezone.utc).timestamp()).encode()).hexdigest()[:16])
    content: str = Field(..., description="Full text content of the document")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata such as author, title, tags")
    source_uri: str | None = Field(default=None, description="Origin file path or URL")
    content_hash: str = Field(default="", description="Fast content hash for deduplication")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def model_post_init(self, __context: Any) -> None:
        if not self.content_hash and self.content:
            self.content_hash = compute_content_hash(self.content)


class ChunkMetadata(BaseModel):
    """Metadata tracking chunk provenance, structural position, and context."""

    doc_id: str
    chunk_index: int
    total_chunks: int = 1
    start_char: int = 0
    end_char: int = 0
    token_count: int = 0
    section_hierarchy: list[str] = Field(default_factory=list, description="Header breadcrumbs: e.g. ['Doc', 'Section 1']")
    context_summary: str | None = Field(default=None, description="Anthropic-style situational context")
    source_uri: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """Atomic retrievable unit with text, metadata, and optional embeddings."""

    id: str = Field(..., description="Unique chunk identifier, e.g. doc123#0")
    text: str = Field(..., description="Raw text of the chunk")
    contextualized_text: str | None = Field(
        default=None,
        description="Chunk text prepended with document/section context for dense/sparse indexing",
    )
    metadata: ChunkMetadata
    embedding: list[float] | None = Field(default=None, description="Dense vector embedding")
    sparse_vector: dict[str, float] | None = Field(default=None, description="Lexical weights e.g. BM25 / SPLADE")

    @property
    def searchable_text(self) -> str:
        """Returns the text that should be indexed by vector DBs and BM25."""
        return self.contextualized_text if self.contextualized_text else self.text


class IngestConfig(BaseModel):
    """Configuration for ingestion and chunking execution."""

    chunk_size: int = Field(default=500, description="Target chunk size in tokens")
    chunk_overlap: int = Field(default=50, description="Token overlap between consecutive chunks")
    tokenizer_name: str = Field(default="cl100k_base", description="Tiktoken or HF tokenizer encoding")
    enable_deduplication: bool = Field(default=True, description="Filter duplicate documents and chunks")
    enable_contextual_enrichment: bool = Field(default=False, description="Enrich chunks with contextual awareness")
