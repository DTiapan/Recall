"""Tests for core data models and protocol implementations."""

import pytest
from recall.core.models import Chunk, ChunkMetadata, Document, IngestConfig, compute_content_hash
from recall.core.interfaces import BaseChunker, BaseDocumentLoader, BaseDeduplicator


def test_document_creation_and_hash():
    doc = Document(content="Production enterprise RAG kit at 10M scale.")
    assert doc.id is not None
    assert doc.content_hash != ""
    assert doc.content_hash == compute_content_hash("Production enterprise RAG kit at 10M scale.")


def test_document_deterministic_hash_across_whitespace():
    # compute_content_hash strips leading/trailing whitespace
    hash1 = compute_content_hash("Hello World")
    hash2 = compute_content_hash("  Hello World \n")
    assert hash1 == hash2


def test_chunk_metadata_and_searchable_text():
    meta = ChunkMetadata(
        doc_id="doc-001",
        chunk_index=0,
        total_chunks=3,
        section_hierarchy=["Executive Summary", "Q3 Performance"],
        context_summary="From SEC 10-Q report of Acme Corp Q3 2023.",
    )
    chunk = Chunk(
        id="doc-001#0",
        text="Revenue increased 15% year-over-year.",
        contextualized_text="From SEC 10-Q report of Acme Corp Q3 2023. Revenue increased 15% year-over-year.",
        metadata=meta,
    )
    assert chunk.searchable_text.startswith("From SEC 10-Q")
    assert chunk.metadata.section_hierarchy == ["Executive Summary", "Q3 Performance"]


def test_ingest_config_defaults():
    config = IngestConfig()
    assert config.chunk_size == 500
    assert config.chunk_overlap == 50
    assert config.tokenizer_name == "cl100k_base"
    assert config.enable_deduplication is True


def test_protocol_conformance():
    class DummyLoader:
        def load(self, source: str) -> list[Document]:
            return [Document(content="dummy")]

    class DummyChunker:
        def chunk(self, document: Document, config: IngestConfig | None = None) -> list[Chunk]:
            return []

    loader = DummyLoader()
    chunker = DummyChunker()
    assert isinstance(loader, BaseDocumentLoader)
    assert isinstance(chunker, BaseChunker)
