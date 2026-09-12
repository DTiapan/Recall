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
    assert report.hit_rate_at_5_rerank >= report.hit_rate_at_5 or report.hit_rate_at_5_rerank >= 0.5
    assert report.latency_p50_ms >= 0.0
    assert report.rerank_latency_p50_ms >= 0.0
