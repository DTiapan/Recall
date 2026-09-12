"""Integration and unit tests for concurrent HybridRetriever with RRF and Circuit Breaker."""

import asyncio
import time
import pytest

from recall.core.interfaces import BaseHybridRetriever
from recall.core.models import Chunk, ChunkMetadata, SearchResult
from recall.embeddings import MockEmbeddingProvider
from recall.retrieval import BM25Index, HybridRetriever
from recall.storage import QdrantVectorStore


@pytest.fixture
def sample_corpus():
    return [
        Chunk(
            id="c1",
            text="PostgreSQL database replication guide and high availability cluster setup.",
            metadata=ChunkMetadata(doc_id="d1", chunk_index=0, source_uri="pg-ha.md", extra={"tech": "postgres"}),
        ),
        Chunk(
            id="c2",
            text="Zero-day security vulnerability CVE-2024-3094 discovered in upstream xz package.",
            metadata=ChunkMetadata(doc_id="d2", chunk_index=0, source_uri="cve.md", extra={"tech": "security"}),
        ),
        Chunk(
            id="c3",
            text="Kubernetes ingress controller configuration with TLS certificates.",
            metadata=ChunkMetadata(doc_id="d3", chunk_index=0, source_uri="k8s.md", extra={"tech": "k8s"}),
        ),
    ]


@pytest.mark.asyncio
async def test_hybrid_retriever_protocol_and_fusion(sample_corpus):
    # Setup mock embedding provider (8 dims)
    embedder = MockEmbeddingProvider(dimensions=8)
    vector_store = QdrantVectorStore(location=":memory:")
    collection_name = "test_hybrid"
    vector_store.create_collection(collection_name, vector_size=8)

    # Embed chunks and index into Qdrant
    embeddings = embedder.embed_texts([c.searchable_text for c in sample_corpus])
    for chunk, emb in zip(sample_corpus, embeddings):
        chunk.embedding = emb
    vector_store.upsert(collection_name, sample_corpus)

    # Index into BM25
    sparse_index = BM25Index()
    sparse_index.index(sample_corpus)

    retriever = HybridRetriever(
        vector_store=vector_store,
        embedding_provider=embedder,
        sparse_index=sparse_index,
        collection_name=collection_name,
        dense_weight=0.5,
        sparse_weight=0.5,
        rrf_k=60,
    )

    assert isinstance(retriever, BaseHybridRetriever)

    # Search for CVE code (strong sparse signal + dense match)
    results = await retriever.retrieve("CVE-2024-3094 xz vulnerability", limit=2)
    assert len(results) >= 1
    assert results[0].chunk_id == "c2"
    assert results[0].score > 0.0
    # Both channels contributed
    assert "sparse" in results[0].vector_name


@pytest.mark.asyncio
async def test_hybrid_retriever_circuit_breaker_on_dense_timeout(sample_corpus):
    """Verifies that if the dense vector store / embedder exceeds the circuit-breaker
    timeout, the retriever does not hang or raise, but gracefully returns sparse results.
    """
    class SlowEmbeddingProvider:
        dimensions = 8
        def embed_query(self, query: str) -> list[float]:
            time.sleep(0.5)  # Block longer than dense_timeout_seconds
            return [0.1] * 8
        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return [[0.1] * 8 for _ in texts]

    vector_store = QdrantVectorStore(location=":memory:")
    collection_name = "test_cb_timeout"
    vector_store.create_collection(collection_name, vector_size=8)

    sparse_index = BM25Index()
    sparse_index.index(sample_corpus)

    # Configure circuit breaker with very short 50ms timeout
    retriever = HybridRetriever(
        vector_store=vector_store,
        embedding_provider=SlowEmbeddingProvider(),
        sparse_index=sparse_index,
        collection_name=collection_name,
        dense_timeout_seconds=0.05,
    )

    start = time.perf_counter()
    results = await retriever.retrieve("PostgreSQL database", limit=2)
    duration = time.perf_counter() - start

    # Duration should be bounded by dense_timeout_seconds (~0.05s) plus small thread overhead,
    # certainly under 0.4s
    assert duration < 0.4
    # Despite dense timing out, sparse BM25 successfully returned PostgreSQL match
    assert len(results) >= 1
    assert results[0].chunk_id == "c1"
    assert results[0].vector_name == "sparse"


@pytest.mark.asyncio
async def test_hybrid_retriever_circuit_breaker_on_vector_store_error(sample_corpus):
    """Verifies that if vector store encounters an unhandled exception (e.g. connection lost),
    the retriever falls back to sparse results.
    """
    class FailingVectorStore:
        def search(self, *args, **kwargs):
            raise ConnectionError("Qdrant cluster unreachable")

    embedder = MockEmbeddingProvider(dimensions=8)
    sparse_index = BM25Index()
    sparse_index.index(sample_corpus)

    retriever = HybridRetriever(
        vector_store=FailingVectorStore(),
        embedding_provider=embedder,
        sparse_index=sparse_index,
        collection_name="broken_collection",
        dense_timeout_seconds=0.5,
    )

    # Retrieval should succeed without raising ConnectionError
    results = await retriever.retrieve("Kubernetes controller TLS", limit=2)
    assert len(results) >= 1
    assert results[0].chunk_id == "c3"
    assert results[0].vector_name == "sparse"


@pytest.mark.asyncio
async def test_hybrid_retriever_score_threshold_filter(sample_corpus):
    embedder = MockEmbeddingProvider(dimensions=8)
    vector_store = QdrantVectorStore(location=":memory:")
    vector_store.create_collection("test_threshold", vector_size=8)
    sparse_index = BM25Index()
    sparse_index.index(sample_corpus)

    retriever = HybridRetriever(
        vector_store=vector_store,
        embedding_provider=embedder,
        sparse_index=sparse_index,
        collection_name="test_threshold",
    )

    # Unrealistic high threshold should filter out all results
    results = await retriever.retrieve("Kubernetes", limit=5, score_threshold=999.0)
    assert len(results) == 0
