"""End-to-End Storage Integration Test.

Validates the full chain:
Enterprise Document -> Format-Aware Ingestion -> FastEmbed Embedding -> QdrantVectorStore -> Semantic Retrieval.
"""

from pathlib import Path
from recall.adapters import ChunkingAdapterRegistry
from recall.core.config import load_config
from recall.core.models import IngestConfig
from recall.embeddings.fastembed_provider import FastEmbedProvider
from recall.storage.qdrant import QdrantVectorStore


def test_end_to_end_storage_and_semantic_retrieval(tmp_path: Path):
    # 1. Load system config
    cfg = load_config()
    collection_name = "enterprise_test_kb"

    # 2. Initialize components
    embedder = FastEmbedProvider(model_name="BAAI/bge-small-en-v1.5")
    vector_store = QdrantVectorStore(location=":memory:")

    vector_store.create_collection(
        collection_name=collection_name,
        vector_size=embedder.dimensions,
        distance="Cosine",
        enable_quantization=True,
    )

    # 3. Create a realistic enterprise HR policy document with a markdown table
    doc_path = tmp_path / "it_security_compliance.md"
    doc_path.write_text(
        "---\n"
        "title: IT Security & Remote Work Policy\n"
        "department: Information Security\n"
        "version: 3.0\n"
        "---\n\n"
        "# Remote Access Requirements\n\n"
        "All remote employees must connect to the corporate network via WireGuard VPN with multi-factor authentication.\n\n"
        "## Password & Token Rotation Schedule\n\n"
        "| Asset Category | Max Lifetime | Min Length | Complexity |\n"
        "|---|---|---|---|\n"
        "| Production Root Credentials | 30 Days | 24 Chars | High Entropy |\n"
        "| SSH User Access Keys | 90 Days | 4096-bit | Ed25519-SK |\n"
        "| API Service Tokens | 180 Days | 64 Chars | Base64 Hex |\n\n"
        "# Incident Response\n\n"
        "Any suspected unauthorized access must be reported immediately to security@enterprise.corp within 15 minutes.\n",
        encoding="utf-8",
    )

    # 4. Ingest and chunk document using format-aware adapter
    registry = ChunkingAdapterRegistry()
    chunks = registry.process(doc_path, config=IngestConfig(chunk_size=300))
    assert len(chunks) >= 2

    # 5. Generate FastEmbed embeddings for chunks
    texts_to_embed = [c.searchable_text for c in chunks]
    embeddings = embedder.embed_texts(texts_to_embed)
    assert len(embeddings) == len(chunks)

    for chunk, emb in zip(chunks, embeddings):
        chunk.embedding = emb

    # 6. Upsert into Qdrant
    inserted_count = vector_store.upsert(collection_name, chunks)
    assert inserted_count == len(chunks)
    assert vector_store.count(collection_name) == len(chunks)

    # 7. Query: Semantic question about password and token lifetime
    query_text = "How often should production root credentials and SSH keys be rotated?"
    query_vec = embedder.embed_query(query_text)

    search_results = vector_store.search(
        collection_name=collection_name,
        query_vector=query_vec,
        limit=2,
    )

    assert len(search_results) > 0
    top_result = search_results[0]

    # Verify retrieval precision: Top result should contain the rotation table or policy
    assert "Rotation Schedule" in top_result.text or "Production Root Credentials" in top_result.text
    assert top_result.score > 0.6
    assert top_result.metadata.policy_name == "IT Security & Remote Work Policy"
    assert top_result.metadata.file_type == "markdown"

    # 8. Query: Filtered query (only return table chunks)
    table_results = vector_store.search(
        collection_name=collection_name,
        query_vector=query_vec,
        limit=2,
        filter_dict={"content_type": "table"},
    )
    assert len(table_results) == 1
    assert table_results[0].metadata.content_type == "table"
    assert "| Production Root Credentials" in table_results[0].text
