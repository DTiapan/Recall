from pathlib import Path
import pytest

from recall.adapters import MarkdownChunkingAdapter
from recall.core.models import IngestConfig
from recall.embeddings import MockEmbeddingProvider
from recall.rerank import ContextCompressor, MockReranker
from recall.retrieval import BM25Index, HybridRetriever
from recall.storage import QdrantVectorStore
from recall.synthesis import Synthesizer


@pytest.mark.asyncio
async def test_full_rag_pipeline_end_to_end(tmp_path: Path):
    # 1. Ingestion Phase: Ingest Markdown document with structural sections
    doc_text = (
        "---\n"
        "policy: Enterprise Authentication Policy\n"
        "department: Infosec\n"
        "---\n"
        "# Enterprise Authentication Guide\n\n"
        "## Multi-Factor Authentication\n"
        "All engineers must configure hardware security keys (FIDO2/WebAuthn) for production access. "
        "SMS-based 2FA is strictly deprecated due to SIM-swapping vulnerabilities.\n\n"
        "## Session Management\n"
        "Production JWT access tokens expire after fifteen minutes. "
        "Refresh tokens rotate on every exchange and are invalidated upon security alerts."
    )
    doc_file = tmp_path / "auth-policy.md"
    doc_file.write_text(doc_text, encoding="utf-8")

    adapter = MarkdownChunkingAdapter()
    chunks = adapter.chunk(doc_file, config=IngestConfig(chunk_size=40, chunk_overlap=5))
    assert len(chunks) >= 2

    # 2. Storage & Vector Indexing Phase
    embedder = MockEmbeddingProvider(dimensions=16)
    vector_store = QdrantVectorStore(location=":memory:")
    collection_name = "e2e_rag_test"
    vector_store.create_collection(collection_name, vector_size=16)

    embeddings = embedder.embed_texts([c.searchable_text for c in chunks])
    for chunk, emb in zip(chunks, embeddings):
        chunk.embedding = emb
    vector_store.upsert(collection_name, chunks)

    sparse_index = BM25Index()
    sparse_index.index(chunks)

    # 3. Hybrid Retrieval & RRF Phase
    retriever = HybridRetriever(
        vector_store=vector_store,
        embedding_provider=embedder,
        sparse_index=sparse_index,
        collection_name=collection_name,
        dense_weight=0.6,
        sparse_weight=0.4,
        rrf_k=60,
    )

    candidates = await retriever.retrieve(
        query="What hardware keys are required for multi-factor authentication?",
        limit=5,
    )
    assert len(candidates) >= 1

    # 4. Reranking & Quality Filtering Phase
    reranker = MockReranker()
    reranked = reranker.rerank(
        query="hardware security keys FIDO2 WebAuthn authentication",
        candidates=candidates,
        top_k=3,
        score_threshold=0.1,
    )
    assert len(reranked) >= 1

    # Context Compression
    compressor = ContextCompressor()
    compressed = compressor.compress(
        query="hardware security keys FIDO2 WebAuthn",
        candidates=reranked,
        max_tokens_per_chunk=80,
    )
    assert len(compressed) >= 1

    # 5. Synthesis & Citation Attribution Phase
    mock_llm_answer = (
        "Engineers are required to configure hardware security keys such as FIDO2 and WebAuthn "
        "for production access [Doc 1]. SMS 2FA is no longer permitted [Doc 1]."
    )
    synthesizer = Synthesizer(
        model_name="mock-e2e-model",
        mock_response=mock_llm_answer,
    )

    response = await synthesizer.synthesize(
        query="What hardware keys are required for multi-factor authentication?",
        candidates=compressed,
    )

    assert response.model_name == "mock-e2e-model"
    assert "FIDO2" in response.answer
    assert len(response.citations) == 1
    assert response.citations[0].doc_index == 1
    assert "auth-policy.md" in response.citations[0].source_uri
    assert len(response.unverified_citations) == 0
