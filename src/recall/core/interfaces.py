"""Protocols and Abstract Base Classes defining plug-and-play RAG component contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

from recall.core.models import Chunk, Document, IngestConfig, SearchResult


@runtime_checkable
class BaseDocumentLoader(Protocol):
    """Protocol for reading raw files/sources into standard Document objects."""

    def load(self, source: str) -> list[Document]:
        """Loads documents from a file path, directory, or URI."""
        ...


@runtime_checkable
class BaseChunker(Protocol):
    """Protocol for chunking a Document into smaller retrievable Chunks."""

    def chunk(self, document: Document, config: IngestConfig | None = None) -> list[Chunk]:
        """Splits a document into chunks respecting token budgets and structural boundaries."""
        ...


@runtime_checkable
class BaseDeduplicator(Protocol):
    """Protocol for deduplicating documents or chunks."""

    def is_duplicate(self, document: Document) -> bool:
        """Returns True if the document has already been ingested or is a near-duplicate."""
        ...

    def register(self, document: Document) -> None:
        """Registers the document's content hash/fingerprint in the deduplication registry."""
        ...


class BaseContextEnricher(ABC):
    """Abstract base class for situational/contextual chunk enrichment."""

    @abstractmethod
    async def enrich_chunks(self, document: Document, chunks: list[Chunk]) -> list[Chunk]:
        """Enriches chunks with contextual summaries or document hierarchy breadcrumbs."""
        pass


@runtime_checkable
class BaseEmbeddingProvider(Protocol):
    """Protocol for generating vector embeddings."""

    @property
    def dimensions(self) -> int:
        """Dimensionality of the dense vector representation."""
        ...

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generates dense vector embeddings for a batch of texts."""
        ...

    def embed_query(self, query: str) -> list[float]:
        """Generates a dense vector embedding for a single query."""
        ...


@runtime_checkable
class BaseVectorStore(Protocol):
    """Protocol for vector storage, indexing, and similarity search."""

    def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: str = "Cosine",
        enable_quantization: bool = True,
    ) -> None:
        """Creates and configures a collection with HNSW index and optional quantization."""
        ...

    def collection_exists(self, collection_name: str) -> bool:
        """Checks if a collection exists."""
        ...

    def upsert(self, collection_name: str, chunks: list[Chunk]) -> int:
        """Upserts chunks into the vector store. Returns number of points inserted."""
        ...

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 10,
        filter_dict: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        """Performs dense vector similarity search with optional metadata filtering."""
        ...

    def count(self, collection_name: str) -> int:
        """Returns total vector count in the collection."""
        ...

    def delete_collection(self, collection_name: str) -> None:
        """Deletes a collection."""
        ...

