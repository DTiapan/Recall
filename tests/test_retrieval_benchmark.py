"""Tests for real-world retrieval benchmark datasets and runner."""

from pathlib import Path

import pytest

from recall.api.service import RAGService
from recall.core.config import AppConfig, EnvSettings, PipelineConfig
from recall.embeddings import MockEmbeddingProvider
from recall.eval.datasets.local import load_local_benchmark
from recall.eval.retrieval_benchmark import RetrievalBenchmarkRunner, resolve_dataset
from recall.rerank import MockReranker
from recall.storage import QdrantVectorStore
from recall.synthesis import Synthesizer

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_ROOT = REPO_ROOT / "data" / "sample"


def test_load_local_sample_dataset():
    corpus = load_local_benchmark(SAMPLE_ROOT, name="sample")
    assert corpus.name == "sample"
    assert len(corpus.documents) == 5
    assert len(corpus.queries) == 10
    assert any("CVE-2024-3094" in doc.text for doc in corpus.documents)
    formats = {doc.metadata.get("format") for doc in corpus.documents}
    assert formats == {"docx", "md", "pdf"}


def test_resolve_dataset_sample():
    corpus = resolve_dataset(dataset="sample", dataset_path=None, limit=None, scale=None)
    assert corpus.name == "sample"
    assert len(corpus.documents) >= 3


def test_scale_to_limit_and_beir_name_suffix():
    from recall.eval.retrieval_benchmark import _scale_to_limit

    assert _scale_to_limit("10k") == 10_000
    assert _scale_to_limit("100k") == 100_000


def test_beir_doc_id_preserved_through_ingest_metadata():
    from recall.eval.retrieval_benchmark import _is_hit
    from recall.eval.datasets.models import BenchmarkQuery
    from recall.core.models import ChunkMetadata, SearchResult

    query = BenchmarkQuery(
        query_id="beir-q1",
        query="example",
        relevant_doc_ids=["3171584"],
        relevant_sources=["beir:scifact/3171584"],
    )
    hit = _is_hit(
        [
            SearchResult(
                chunk_id="c1",
                text="Scientific abstract text.",
                score=0.9,
                metadata=ChunkMetadata(
                    doc_id="3171584",
                    chunk_index=0,
                    source_uri="beir:scifact/3171584",
                ),
            )
        ],
        query,
        5,
    )
    assert hit is True


@pytest.mark.asyncio
async def test_retrieval_benchmark_on_real_sample_corpus():
    corpus = load_local_benchmark(SAMPLE_ROOT, name="sample")
    service = RAGService(
        config=AppConfig(
            env=EnvSettings(rag_env="test", rag_mode="local"),
            pipeline=PipelineConfig(),
        ),
        vector_store=QdrantVectorStore(location=":memory:"),
        embedding_provider=MockEmbeddingProvider(dimensions=16),
        reranker=MockReranker(),
        synthesizer=Synthesizer(model_name="mock", mock_response="benchmark [Doc 1]."),
        default_collection="bench_real_sample",
    )

    runner = RetrievalBenchmarkRunner(service=service)
    report = await runner.run(corpus=corpus, collection_name="bench_real_sample")

    assert report.documents_ingested == 5
    assert report.queries_evaluated == 10
    assert report.hit_rate_at_5 >= 0.5
    assert report.mrr > 0.0
    assert report.latency_p50_ms >= 0.0


@pytest.mark.asyncio
async def test_retrieval_benchmark_reports_rerank_metrics_when_enabled():
    corpus = load_local_benchmark(SAMPLE_ROOT, name="sample")
    service = RAGService(
        config=AppConfig(
            env=EnvSettings(rag_env="test", rag_mode="local"),
            pipeline=PipelineConfig(),
        ),
        vector_store=QdrantVectorStore(location=":memory:"),
        embedding_provider=MockEmbeddingProvider(dimensions=16),
        reranker=MockReranker(),
        synthesizer=Synthesizer(model_name="mock", mock_response="benchmark [Doc 1]."),
        default_collection="bench_rerank_metrics",
    )
    service.config.pipeline.reranking.candidate_k = 12

    runner = RetrievalBenchmarkRunner(service=service)
    report = await runner.run(
        corpus=corpus,
        collection_name="bench_rerank_metrics",
        include_rerank=True,
        retrieve_limit=12,
    )

    assert report.rerank_evaluated is True
    assert report.rerank_candidate_k == 12
    assert report.hit_rate_at_5_rerank >= 0.0
    assert report.mrr_rerank >= 0.0
    assert "rerank" in report.to_markdown().lower()


@pytest.mark.asyncio
async def test_retrieval_benchmark_reuses_persisted_index(tmp_path):
    from recall.eval.benchmark_index import build_index_fingerprint, resolve_benchmark_index

    corpus = load_local_benchmark(SAMPLE_ROOT, name="sample")
    index = resolve_benchmark_index(
        dataset="sample",
        document_limit=None,
        scale=None,
        subsample_seed=42,
        fast=False,
        index_dir=tmp_path,
    )
    service = RAGService(
        config=AppConfig(
            env=EnvSettings(rag_env="test", rag_mode="local"),
            pipeline=PipelineConfig(),
        ),
        vector_store=QdrantVectorStore(path=str(index.qdrant_path)),
        embedding_provider=MockEmbeddingProvider(dimensions=16),
        reranker=MockReranker(),
        synthesizer=Synthesizer(model_name="mock", mock_response="benchmark [Doc 1]."),
        default_collection=index.collection_name,
    )

    runner = RetrievalBenchmarkRunner(service=service)
    fingerprint = build_index_fingerprint(
        dataset="sample",
        document_count=len(corpus.documents),
        subsample_seed=42,
        scale=None,
        document_limit=None,
        dense_model="unknown-dense",
        sparse_model="unknown-sparse",
        chunk_size=500,
        rag_mode="local",
        fast_mode=False,
    )
    first = await runner.run(
        corpus=corpus,
        collection_name=index.collection_name,
        index_context=index,
        index_fingerprint=fingerprint,
        reuse_index=False,
    )

    second = await runner.run(
        corpus=corpus,
        collection_name=index.collection_name,
        index_context=index,
        index_fingerprint=fingerprint,
        reuse_index=True,
    )

    assert first.index_reused is False
    assert second.index_reused is True
    assert second.ingest_seconds < first.ingest_seconds
    assert second.documents_ingested == first.documents_ingested
