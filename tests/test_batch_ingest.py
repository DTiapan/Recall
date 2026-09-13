"""Tests for batched high-throughput chunk ingestion."""

import pytest

from recall.api.service import RAGService
from recall.core.config import AppConfig, EnvSettings, PipelineConfig
from recall.embeddings import MockEmbeddingProvider, MockSparseEmbeddingProvider
from recall.storage import QdrantVectorStore


@pytest.mark.asyncio
async def test_ingest_chunks_batched_upserts_in_micro_batches():
    service = RAGService(
        config=AppConfig(
            env=EnvSettings(rag_env="test", rag_mode="local"),
            pipeline=PipelineConfig(),
        ),
        vector_store=QdrantVectorStore(location=":memory:"),
        embedding_provider=MockEmbeddingProvider(dimensions=16),
        sparse_embedder=MockSparseEmbeddingProvider(),
        default_collection="batch_ingest",
    )

    chunks = [
        service.build_text_chunk(f"Document body {index}", f"doc_{index}.txt", f"doc_{index}")
        for index in range(10)
    ]
    indexed = await service.ingest_chunks_batched(
        chunks,
        collection_name="batch_ingest",
        batch_size=4,
    )

    assert indexed == 10
    assert service.vector_store.count("batch_ingest") == 10
