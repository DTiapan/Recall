"""Embedding gateway interfaces and provider implementations."""

from recall.embeddings.fastembed_provider import FastEmbedProvider
from recall.embeddings.fastembed_sparse_provider import FastEmbedSparseProvider
from recall.embeddings.mock_provider import MockEmbeddingProvider
from recall.embeddings.mock_sparse_provider import MockSparseEmbeddingProvider

__all__ = [
    "FastEmbedProvider",
    "FastEmbedSparseProvider",
    "MockEmbeddingProvider",
    "MockSparseEmbeddingProvider",
]
