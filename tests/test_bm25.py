"""Unit tests for BM25 lexical sparse index."""

import pytest
from recall.core.interfaces import BaseSparseIndex
from recall.core.models import Chunk, ChunkMetadata
from recall.retrieval.bm25 import BM25Index, tokenize_lexical


def test_tokenize_lexical():
    # Technical IDs, CVEs, RFCs, and hyphenated terms
    tokens = tokenize_lexical("Fixed CVE-2024-3094 in liblzma and RFC-4122 UUID generator!")
    assert "cve-2024-3094" in tokens
    assert "rfc-4122" in tokens
    assert "liblzma" in tokens
    assert "uuid" in tokens


def test_bm25_index_protocol_conformance():
    index = BM25Index()
    assert isinstance(index, BaseSparseIndex)
    assert index.count() == 0


def test_bm25_exact_keyword_and_code_search():
    index = BM25Index()
    chunks = [
        Chunk(
            id="c1",
            document_id="d1",
            text="The vulnerability CVE-2024-3094 allows remote SSH authentication bypass.",
            index=0,
            token_count=10,
            metadata=ChunkMetadata(doc_id="d1", chunk_index=0, source_uri="sec-advisory.md", extra={"category": "security"}),
        ),
        Chunk(
            id="c2",
            document_id="d2",
            text="PostgreSQL connection pooling configuration using PgBouncer on port 6432.",
            index=0,
            token_count=10,
            metadata=ChunkMetadata(doc_id="d2", chunk_index=0, source_uri="db-guide.md", extra={"category": "database"}),
        ),
        Chunk(
            id="c3",
            document_id="d3",
            text="Python async/await concurrency patterns using asyncio.gather.",
            index=0,
            token_count=10,
            metadata=ChunkMetadata(doc_id="d3", chunk_index=0, source_uri="python-async.md", extra={"category": "development"}),
        ),
    ]

    indexed_count = index.index(chunks)
    assert indexed_count == 3
    assert index.count() == 3

    # Exact search for CVE code
    results = index.search("CVE-2024-3094", limit=5)
    assert len(results) >= 1
    assert results[0].chunk_id == "c1"
    assert results[0].vector_name == "sparse"
    assert results[0].score > 0.0

    # Search for port number and tool
    results_db = index.search("PgBouncer 6432", limit=5)
    assert len(results_db) >= 1
    assert results_db[0].chunk_id == "c2"


def test_bm25_metadata_filtering():
    index = BM25Index()
    chunks = [
        Chunk(
            id="c1",
            document_id="d1",
            text="Kubernetes deployment spec for production cluster.",
            index=0,
            token_count=8,
            metadata=ChunkMetadata(doc_id="d1", chunk_index=0, source_uri="k8s.yaml", extra={"env": "production"}),
        ),
        Chunk(
            id="c2",
            document_id="d2",
            text="Kubernetes deployment spec for staging and sandbox cluster.",
            index=0,
            token_count=8,
            metadata=ChunkMetadata(doc_id="d2", chunk_index=0, source_uri="k8s-stage.yaml", extra={"env": "staging"}),
        ),
    ]
    index.index(chunks)

    # Search with production filter
    prod_results = index.search("Kubernetes cluster", filter_dict={"env": "production"})
    assert len(prod_results) == 1
    assert prod_results[0].chunk_id == "c1"

    # Search with staging filter
    stage_results = index.search("Kubernetes cluster", filter_dict={"env": "staging"})
    assert len(stage_results) == 1
    assert stage_results[0].chunk_id == "c2"


def test_bm25_empty_and_edge_cases():
    index = BM25Index()
    assert index.search("anything") == []

    # Index empty list
    assert index.index([]) == 0

    chunks = [
        Chunk(
            id="c1",
            document_id="d1",
            text="Simple sample text",
            index=0,
            token_count=4,
            metadata=ChunkMetadata(doc_id="d1", chunk_index=0),
        )
    ]
    index.index(chunks)
    # Search with no matching terms
    assert index.search("xylophone zanzibar") == []
