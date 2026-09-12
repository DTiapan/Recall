"""Protocols and Abstract Base Classes defining plug-and-play RAG component contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

from rag_kit.core.models import Chunk, Document, IngestConfig


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
