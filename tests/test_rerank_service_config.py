"""Tests that RAGService wires reranker config correctly (AP-003)."""

from __future__ import annotations

import pytest

from recall.api.service import RAGService
from recall.core.config import AppConfig, EnvSettings, PipelineConfig
from recall.core.models import ChunkMetadata, SearchResult
from recall.embeddings import MockEmbeddingProvider
from recall.rerank.mock import MockReranker
from recall.storage import QdrantVectorStore
from recall.synthesis import Synthesizer


class _RecordingReranker(MockReranker):
    last_score_threshold: float | None = None
    last_candidate_count: int = 0

    def rerank(self, query, candidates, top_k=5, score_threshold=None):
        self.last_score_threshold = score_threshold
        self.last_candidate_count = len(candidates)
        return super().rerank(query, candidates, top_k=top_k, score_threshold=score_threshold)


def _test_service(reranker: _RecordingReranker) -> RAGService:
    config = AppConfig(
        env=EnvSettings(rag_env="test", rag_mode="local"),
        pipeline=PipelineConfig(),
    )
    return RAGService(
        config=config,
        vector_store=QdrantVectorStore(location=":memory:"),
        embedding_provider=MockEmbeddingProvider(dimensions=8),
        reranker=reranker,
        synthesizer=Synthesizer(model_name="mock", mock_response="ok"),
        default_collection="rerank_config_test",
    )


@pytest.mark.asyncio
async def test_search_rerank_uses_reranking_score_threshold_not_retrieval():
    reranker = _RecordingReranker()
    service = _test_service(reranker)
    service.config.pipeline.retrieval.score_threshold = 0.99
    service.config.pipeline.reranking.score_threshold = None

    candidate = SearchResult(
        chunk_id="c1",
        score=0.9,
        text="PostgreSQL connection pooling with PgBouncer.",
        metadata=ChunkMetadata(doc_id="d1", chunk_index=0),
        vector_name="fused",
    )

    async def _fake_search(*_args, **_kwargs):
        return [candidate]

    service.search = _fake_search  # type: ignore[method-assign]

    await service.search_rerank("PostgreSQL pooling", limit=1, collection_name="test")

    assert reranker.last_score_threshold is None


@pytest.mark.asyncio
async def test_search_rerank_passes_configured_reranking_threshold():
    reranker = _RecordingReranker()
    service = _test_service(reranker)
    service.config.pipeline.reranking.score_threshold = 0.42

    candidate = SearchResult(
        chunk_id="c1",
        score=0.9,
        text="Kubernetes pod autoscaling policy.",
        metadata=ChunkMetadata(doc_id="d1", chunk_index=0),
        vector_name="fused",
    )

    async def _fake_search(*_args, **_kwargs):
        return [candidate]

    service.search = _fake_search  # type: ignore[method-assign]

    await service.search_rerank("Kubernetes autoscaler", limit=1, collection_name="test")

    assert reranker.last_score_threshold == 0.42


@pytest.mark.asyncio
async def test_search_rerank_uses_candidate_k_from_config():
    reranker = _RecordingReranker()
    service = _test_service(reranker)
    service.config.pipeline.reranking.candidate_k = 50

    candidates = [
        SearchResult(
            chunk_id=f"c{i}",
            score=0.9 - i * 0.01,
            text=f"Candidate passage {i}.",
            metadata=ChunkMetadata(doc_id=f"d{i}", chunk_index=0),
            vector_name="fused",
        )
        for i in range(50)
    ]

    async def _fake_search(*_args, limit: int = 10, **_kwargs):
        assert limit == 50
        return candidates

    service.search = _fake_search  # type: ignore[method-assign]

    await service.search_rerank("candidate pool sizing", limit=5, collection_name="test")

    assert reranker.last_candidate_count == 50


@pytest.mark.asyncio
async def test_query_skips_rerank_when_disabled():
    reranker = _RecordingReranker()
    service = _test_service(reranker)
    service.config.pipeline.reranking.enabled = False

    candidate = SearchResult(
        chunk_id="c1",
        score=0.9,
        text="PostgreSQL connection pooling with PgBouncer.",
        metadata=ChunkMetadata(doc_id="d1", chunk_index=0),
        vector_name="fused",
    )

    async def _fake_search(*_args, limit: int = 10, **_kwargs):
        assert limit == 5
        return [candidate]

    service.search = _fake_search  # type: ignore[method-assign]
    service.synthesizer = Synthesizer(model_name="mock", mock_response="ok [Doc 1].")

    response = await service.query("PostgreSQL pooling", collection_name="test", top_k=5)

    assert reranker.last_candidate_count == 0
    assert response.answer == "ok [Doc 1]."
