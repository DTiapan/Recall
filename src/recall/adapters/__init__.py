"""Extensible adapter framework for format-aware and structural document chunking."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

from recall.core.models import Chunk, IngestConfig


class BaseChunkingAdapter(ABC):
    """Abstract base class for format-specific chunking adapters."""

    SUPPORTED_EXTENSIONS: ClassVar[set[str]] = set()

    def supports(self, file_path: Path, mime_type: str | None = None) -> bool:
        """Returns True if this adapter can process the specified file."""
        return file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    @abstractmethod
    def chunk(self, file_path: Path, config: IngestConfig | None = None) -> list[Chunk]:
        """Parses the document structure and returns enriched, format-aware Chunks."""
        pass


class ChunkingAdapterRegistry:
    """Central registry mapping file formats and MIME types to specialized chunking adapters."""

    def __init__(self, register_defaults: bool = True) -> None:
        self._adapters: list[BaseChunkingAdapter] = []
        if register_defaults:
            self._register_default_adapters()

    def _register_default_adapters(self) -> None:
        from recall.adapters.pdf_adapter import PDFChunkingAdapter
        from recall.adapters.docx_adapter import DocxChunkingAdapter
        from recall.adapters.markdown_adapter import MarkdownChunkingAdapter
        from recall.adapters.text_adapter import TextChunkingAdapter

        self.register(TextChunkingAdapter())
        self.register(MarkdownChunkingAdapter())
        self.register(DocxChunkingAdapter())
        self.register(PDFChunkingAdapter())

    def register(self, adapter: BaseChunkingAdapter) -> None:
        """Registers a chunking adapter (newly registered adapters take precedence)."""
        self._adapters.insert(0, adapter)

    def get_adapter(self, file_path: str | Path) -> BaseChunkingAdapter:
        """Finds the first adapter that supports the file format."""
        path = Path(file_path)
        for adapter in self._adapters:
            if adapter.supports(path):
                return adapter
        raise ValueError(f"No chunking adapter registered for file: {path.name} ({path.suffix})")

    def process(self, file_path: str | Path, config: IngestConfig | None = None) -> list[Chunk]:
        """Looks up the appropriate adapter and processes the file."""
        adapter = self.get_adapter(file_path)
        return adapter.chunk(Path(file_path), config)


from recall.adapters.docx_adapter import DocxChunkingAdapter
from recall.adapters.markdown_adapter import MarkdownChunkingAdapter
from recall.adapters.pdf_adapter import PDFChunkingAdapter
from recall.adapters.text_adapter import TextChunkingAdapter

__all__ = [
    "BaseChunkingAdapter",
    "ChunkingAdapterRegistry",
    "DocxChunkingAdapter",
    "MarkdownChunkingAdapter",
    "PDFChunkingAdapter",
    "TextChunkingAdapter",
]
