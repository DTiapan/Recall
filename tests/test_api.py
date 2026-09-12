"""Integration tests for Recall FastAPI REST endpoints and Web UI."""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from recall.api.app import create_app
from recall.api.service import RAGService
from recall.core.config import AppConfig, EnvSettings, PipelineConfig
from recall.embeddings import MockEmbeddingProvider
from recall.rerank import MockReranker
from recall.storage import QdrantVectorStore
from recall.synthesis import Synthesizer


@pytest.fixture
def mock_service():
    embedder = MockEmbeddingProvider(dimensions=8)
    vector_store = QdrantVectorStore(location=":memory:")
    reranker = MockReranker()
    synthesizer = Synthesizer(
        model_name="mock-model",
        mock_response="PostgreSQL replication uses streaming replication [Doc 1].",
    )
    config = AppConfig(
        env=EnvSettings(rag_env="test", rag_mode="local"),
        pipeline=PipelineConfig(),
    )
    return RAGService(
        config=config,
        vector_store=vector_store,
        embedding_provider=embedder,
        reranker=reranker,
        synthesizer=synthesizer,
        default_collection="test_api_col",
    )


@pytest.fixture
def client(mock_service):
    app = create_app(rag_service=mock_service)
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "recall-kit"
    assert data["embedding_dimensions"] == 8


def test_root_index_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Recall" in response.text
    assert "Ask Recall Anything" in response.text


def test_ingest_text_and_search(client):
    # Ingest text
    ingest_res = client.post(
        "/v1/ingest",
        data={"text": "PostgreSQL replication uses streaming replication for high availability.", "source_uri": "pg.md"},
    )
    assert ingest_res.status_code == 200
    assert ingest_res.json()["chunks_indexed"] == 1

    # Search
    search_res = client.post(
        "/v1/search",
        json={"query": "PostgreSQL replication streaming", "limit": 5},
    )
    assert search_res.status_code == 200
    results = search_res.json()
    assert len(results) >= 1
    assert "PostgreSQL" in results[0]["text"]


def test_ingest_file_upload(client, tmp_path: Path):
    test_file = tmp_path / "k8s.txt"
    test_file.write_text("Kubernetes cluster networking uses CNI plugins such as Cilium or Calico.")

    with open(test_file, "rb") as f:
        ingest_res = client.post(
            "/v1/ingest",
            files={"file": ("k8s.txt", f, "text/plain")},
        )
    assert ingest_res.status_code == 200
    assert ingest_res.json()["chunks_indexed"] >= 1


def test_chat_pipeline_endpoint(client):
    # Ingest source document
    client.post(
        "/v1/ingest",
        data={"text": "PostgreSQL replication uses streaming replication.", "source_uri": "pg.md"},
    )

    # Chat query
    chat_res = client.post(
        "/v1/chat",
        json={"query": "How does PostgreSQL replication work?"},
    )
    assert chat_res.status_code == 200
    data = chat_res.json()
    assert "streaming replication" in data["answer"]
    assert len(data["citations"]) == 1
    assert data["citations"][0]["doc_index"] == 1
    assert data["citations"][0]["source_uri"] == "pg.md"


def test_stats_endpoint(client):
    res = client.get("/v1/stats")
    assert res.status_code == 200
    data = res.json()
    assert "total_chunks" in data
