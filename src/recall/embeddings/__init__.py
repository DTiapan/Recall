"""Embedding gateway interfaces and provider implementations."""

from recall.embeddings.fastembed_provider import FastEmbedProvider
from recall.embeddings.mock_provider import MockEmbeddingProvider

__all__ = [
    "FastEmbedProvider",
    "MockEmbeddingProvider",
]
