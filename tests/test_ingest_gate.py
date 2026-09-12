"""Tests for live ingest deduplication gates."""

from pathlib import Path

import pytest

from recall.api.service import RAGService
from recall.core.config import AppConfig, EnvSettings, PipelineConfig
from recall.core.models import Document
from recall.embeddings import MockEmbeddingProvider
from recall.preprocessing.ingest_gate import DedupGateRegistry, should_ingest_document
from recall.rerank import MockReranker
from recall.storage import QdrantVectorStore
from recall.synthesis import Synthesizer


@pytest.mark.asyncio
async def test_service_skips_exact_duplicate_file_ingest(tmp_path: Path):
    doc_path = tmp_path / "policy.md"
    doc_path.write_text("# Policy\nAll staff must use hardware MFA.", encoding="utf-8")

    service = _build_service()
    first = await service.ingest_file(doc_path)
    second = await service.ingest_file(doc_path)

    assert first >= 1
    assert second == 0


@pytest.mark.asyncio
async def test_service_skips_near_duplicate_text_ingest():
    service = _build_service()
    base = (
        "All engineers must configure hardware security keys for production access. "
        "SMS-based two factor authentication is strictly deprecated due to SIM swapping risks. "
        "Production JWT access tokens expire after fifteen minutes and refresh tokens rotate "
        "on every exchange and are invalidated upon security alerts."
    )

    first = await service.ingest_text(base, source_uri="auth-policy.md")
    second = await service.ingest_text(base + " Minor footnote update.", source_uri="auth-policy-v2.md")

    assert first == 1
    assert second == 0


def test_should_ingest_document_registers_unique_documents():
    registry = DedupGateRegistry(near_dup_threshold=0.85)
    gate = registry.get("docs")
    document = Document(content="Unique onboarding checklist for new hires.", source_uri="onboarding.md")

    assert should_ingest_document(document, gate, enable_deduplication=True) is True
    assert should_ingest_document(document, gate, enable_deduplication=True) is False


def _build_service() -> RAGService:
    config = AppConfig(
        env=EnvSettings(rag_env="test", rag_mode="local"),
        pipeline=PipelineConfig(),
    )
    return RAGService(
        config=config,
        vector_store=QdrantVectorStore(location=":memory:"),
        embedding_provider=MockEmbeddingProvider(dimensions=8),
        reranker=MockReranker(),
        synthesizer=Synthesizer(model_name="mock", mock_response="Answer [Doc 1]."),
        default_collection="dedup_test",
    )
