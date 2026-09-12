"""Contextual awareness chunker injecting document and section hierarchy context into chunks."""

from __future__ import annotations

from typing import Callable
from recall.core.interfaces import BaseChunker
from recall.core.models import Chunk, Document, IngestConfig
from recall.chunkers.recursive import RecursiveChunker


class ContextualChunker(BaseChunker):
    """Augments chunk embeddings and sparse indices with document-level and section-level context.
    
    Addresses the 'lost context' failure mode where individual chunks contain ambiguous
    pronouns, unqualified metrics ('revenue grew 3%'), or disconnected table rows.
    """

    def __init__(
        self,
        base_chunker: BaseChunker | None = None,
        context_generator: Callable[[Document, Chunk], str] | None = None,
    ) -> None:
        self.base_chunker = base_chunker or RecursiveChunker()
        self.context_generator = context_generator or self._default_context_generator

    @staticmethod
    def _default_context_generator(document: Document, chunk: Chunk) -> str:
        """Constructs a deterministic situational header from document title and structural headings."""
        parts: list[str] = []
        title = document.metadata.get("title") or document.metadata.get("filename")
        if title:
            parts.append(f"Document: {title}")

        headings = document.metadata.get("headings", [])
        if headings:
            parts.append(f"Sections: {' > '.join(headings[:3])}")

        if parts:
            return f"[{'; '.join(parts)}]\n"
        return ""

    def chunk(self, document: Document, config: IngestConfig | None = None) -> list[Chunk]:
        base_chunks = self.base_chunker.chunk(document, config)

        for chunk in base_chunks:
            context_prefix = self.context_generator(document, chunk)
            if context_prefix:
                chunk.contextualized_text = f"{context_prefix}{chunk.text}"
                chunk.metadata.context_summary = context_prefix.strip()
            else:
                chunk.contextualized_text = chunk.text

        return base_chunks
