"""Tests for QdrantVectorStore in-memory indexing, quantization, and search."""

import numpy as np
import pytest
from recall.core.interfaces import BaseVectorStore
from recall.core.models import Chunk, ChunkMetadata
from recall.embeddings.mock_provider import MockEmbeddingProvider
from recall.storage.qdrant import QdrantVectorStore


def test_qdrant_vector_store_lifecycle_and_search():
    store = QdrantVectorStore(location=":memory:")
    assert isinstance(store, BaseVectorStore)

    col_name = "test_enterprise_kb"
    dim = 64
    embedder = MockEmbeddingProvider(dimensions=dim)

    # 1. Collection creation
    assert not store.collection_exists(col_name)
    store.create_collection(
        collection_name=col_name,
        vector_size=dim,
        distance="Cosine",
        enable_quantization=True,
    )
    assert store.collection_exists(col_name)
    assert store.count(col_name) == 0

    # 2. Build test chunks
    c1 = Chunk(
        id="doc1#0",
        text="All production database credentials must be rotated every 90 days.",
        metadata=ChunkMetadata(
            doc_id="doc1",
            chunk_index=0,
            file_type="markdown",
            policy_name="Security Policy",
            content_type="text",
        ),
        embedding=embedder.embed_query("All production database credentials must be rotated every 90 days."),
    )

    c2 = Chunk(
        id="doc2#0",
        text="Full-time employees receive comprehensive dental and vision healthcare benefits.",
        metadata=ChunkMetadata(
            doc_id="doc2",
            chunk_index=0,
            file_type="pdf",
            policy_name="Benefits Handbook",
            content_type="text",
        ),
        embedding=embedder.embed_query("Full-time employees receive comprehensive dental and vision healthcare benefits."),
    )

    # 3. Upsert
    inserted = store.upsert(col_name, [c1, c2])
    assert inserted == 2
    assert store.count(col_name) == 2

    # 4. Search exact match
    query_vec = embedder.embed_query("All production database credentials must be rotated every 90 days.")
    results = store.search(col_name, query_vector=query_vec, limit=2)
    assert len(results) == 2
    # Nearest neighbor must be c1 with score ~1.0
    assert results[0].chunk_id == "doc1#0"
    assert "credentials" in results[0].text
    assert np.isclose(results[0].score, 1.0, atol=1e-3)
    assert results[0].score > results[1].score
    assert results[0].metadata.policy_name == "Security Policy"

    # 5. Metadata filtering (filter by file_type="pdf")
    filtered_results = store.search(
        col_name,
        query_vector=query_vec,
        limit=2,
        filter_dict={"file_type": "pdf"},
    )
    assert len(filtered_results) == 1
    assert filtered_results[0].chunk_id == "doc2#0"
    assert filtered_results[0].metadata.file_type == "pdf"

    # 6. Delete collection
    store.delete_collection(col_name)
    assert not store.collection_exists(col_name)
