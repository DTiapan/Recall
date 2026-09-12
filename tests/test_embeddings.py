"""Tests for embedding providers (Mock & FastEmbed)."""

import numpy as np
import pytest
from recall.core.interfaces import BaseEmbeddingProvider
from recall.embeddings.fastembed_provider import FastEmbedProvider
from recall.embeddings.mock_provider import MockEmbeddingProvider


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    a = np.array(v1)
    b = np.array(v2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def test_mock_embedding_provider_protocol_and_properties():
    provider = MockEmbeddingProvider(dimensions=128)
    assert isinstance(provider, BaseEmbeddingProvider)
    assert provider.dimensions == 128

    v1 = provider.embed_query("Enterprise search query")
    v2 = provider.embed_query("Enterprise search query")
    v3 = provider.embed_query("Completely unrelated text")

    assert len(v1) == 128
    # Determinism
    assert v1 == v2
    # Distinct texts yield different vectors
    assert v1 != v3
    # Unit norm
    assert np.isclose(np.linalg.norm(v1), 1.0, atol=1e-5)


def test_fastembed_provider_protocol_and_dimensions():
    provider = FastEmbedProvider(model_name="BAAI/bge-small-en-v1.5")
    assert isinstance(provider, BaseEmbeddingProvider)
    assert provider.dimensions == 384

    # Empty input handling
    assert provider.embed_texts([]) == []
    assert len(provider.embed_query("")) == 384

    # Batch embedding
    texts = [
        "Retrieval-Augmented Generation for enterprise data",
        "Vector databases index high-dimensional embeddings",
    ]
    embeddings = provider.embed_texts(texts)
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 384
    assert len(embeddings[1]) == 384


def test_fastembed_semantic_similarity():
    provider = FastEmbedProvider(model_name="BAAI/bge-small-en-v1.5")

    query = provider.embed_query("How to index vectors in Qdrant database")
    doc_relevant = provider.embed_query("Qdrant is a vector database designed for high-dimensional search")
    doc_irrelevant = provider.embed_query("A delicious chocolate chip cookie recipe with butter and sugar")

    sim_rel = cosine_similarity(query, doc_relevant)
    sim_irrel = cosine_similarity(query, doc_irrelevant)

    assert sim_rel > sim_irrel
    assert sim_rel > 0.6
