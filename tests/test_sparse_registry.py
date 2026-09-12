"""Tests for per-collection BM25 index registry."""

from recall.core.models import Chunk, ChunkMetadata
from recall.retrieval.sparse_registry import BM25IndexRegistry, RegistrySparseIndexAdapter


def test_registry_keeps_collections_isolated():
    registry = BM25IndexRegistry()

    registry.index(
        "alpha",
        [
            Chunk(
                id="a1",
                text="Alpha collection discusses WireGuard VPN remote access.",
                metadata=ChunkMetadata(doc_id="d1", chunk_index=0),
            )
        ],
    )
    registry.index(
        "beta",
        [
            Chunk(
                id="b1",
                text="Beta collection discusses Kubernetes ingress controllers.",
                metadata=ChunkMetadata(doc_id="d2", chunk_index=0),
            )
        ],
    )

    alpha_hits = registry.search("alpha", "WireGuard VPN", limit=5)
    beta_hits = registry.search("beta", "WireGuard VPN", limit=5)

    assert len(alpha_hits) == 1
    assert alpha_hits[0].chunk_id == "a1"
    assert beta_hits == []


def test_registry_adapter_implements_sparse_index_protocol():
    registry = BM25IndexRegistry()
    adapter = RegistrySparseIndexAdapter(registry, "docs")

    adapter.index(
        [
            Chunk(
                id="c1",
                text="Adapter path indexes PostgreSQL replication guidance.",
                metadata=ChunkMetadata(doc_id="d1", chunk_index=0),
            )
        ]
    )

    assert adapter.count() == 1
    results = adapter.search("PostgreSQL replication", limit=1)
    assert results[0].chunk_id == "c1"
