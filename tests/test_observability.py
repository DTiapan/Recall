"""Tests for OpenTelemetry tracing bootstrap and span emission."""

import pytest

pytestmark = pytest.mark.xdist_group("otel")
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from recall.api.app import create_app
from recall.api.service import RAGService
from recall.core.config import AppConfig, EnvSettings, PipelineConfig
from recall.embeddings import MockEmbeddingProvider
from recall.observability.tracing import init_tracing, reset_tracing
from recall.rerank import MockReranker
from recall.storage import QdrantVectorStore
from recall.synthesis import Synthesizer


@pytest.fixture(autouse=True)
def reset_tracing_state():
    reset_tracing()
    yield
    reset_tracing()


@pytest.fixture
def memory_exporter() -> InMemorySpanExporter:
    reset_tracing()
    exporter = InMemorySpanExporter()
    init_tracing(service_name="recall-test", enabled=True, extra_exporter=exporter)
    yield exporter
    exporter.clear()
    reset_tracing()


@pytest.fixture
def traced_service(memory_exporter: InMemorySpanExporter):
    embedder = MockEmbeddingProvider(dimensions=8)
    vector_store = QdrantVectorStore(location=":memory:")
    reranker = MockReranker()
    synthesizer = Synthesizer(
        model_name="mock-model",
        mock_response="Observed answer with citation [Doc 1].",
    )
    config = AppConfig(
        env=EnvSettings(rag_env="test", rag_mode="local", otel_sdk_disabled=False),
        pipeline=PipelineConfig(),
    )
    return RAGService(
        config=config,
        vector_store=vector_store,
        embedding_provider=embedder,
        reranker=reranker,
        synthesizer=synthesizer,
        default_collection="trace_test",
    )


def test_health_endpoint_reports_tracing_enabled(traced_service: RAGService):
    app = create_app(rag_service=traced_service)
    client = TestClient(app)

    response = client.get("/v1/health")
    assert response.status_code == 200
    assert response.json()["tracing_enabled"] is True
    assert response.headers.get("x-request-id")


@pytest.mark.asyncio
async def test_query_pipeline_emits_core_spans(traced_service: RAGService, memory_exporter: InMemorySpanExporter):
    await traced_service.ingest_text(
        "OpenTelemetry tracing validates ingest and retrieval spans.",
        source_uri="otel.md",
    )
    await traced_service.query("What does OpenTelemetry validate?")

    span_names = {span.name for span in memory_exporter.get_finished_spans()}
    assert "ingest.text" in span_names
    assert "query.pipeline" in span_names
    assert "retrieve.hybrid" in span_names
    assert "synthesize" in span_names


def test_tracing_can_be_disabled_via_env():
    reset_tracing()
    app = create_app(
        rag_service=RAGService(
            config=AppConfig(
                env=EnvSettings(rag_env="test", rag_mode="local", otel_sdk_disabled=True),
                pipeline=PipelineConfig(),
            ),
            vector_store=QdrantVectorStore(location=":memory:"),
            embedding_provider=MockEmbeddingProvider(dimensions=8),
            reranker=MockReranker(),
            synthesizer=Synthesizer(model_name="mock", mock_response="ok [Doc 1]."),
            default_collection="trace_off",
        )
    )
    client = TestClient(app)
    response = client.get("/v1/health")
    assert response.json()["tracing_enabled"] is False
