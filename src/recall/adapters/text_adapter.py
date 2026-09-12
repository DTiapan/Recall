"""Plain text chunking adapter with recursive paragraph and sentence splitting."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar

from recall.adapters import BaseChunkingAdapter
from recall.core.models import Chunk, ChunkMetadata, IngestConfig
from recall.chunkers.recursive import RecursiveChunker
from recall.loaders.text import TextLoader


class TextChunkingAdapter(BaseChunkingAdapter):
    """Loads and chunks standard text files with encoding fallback and metadata attachment."""

    SUPPORTED_EXTENSIONS: ClassVar[set[str]] = {".txt", ".text", ".log"}

    def __init__(self) -> None:
        self.loader = TextLoader(autodetect_encoding=True)
        self.chunker = RecursiveChunker()

    def chunk(self, file_path: Path, config: IngestConfig | None = None) -> list[Chunk]:
        if not file_path.exists():
            raise FileNotFoundError(f"Text file not found: {file_path}")

        mtime = datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc)
        docs = self.loader.load(str(file_path))
        if not docs:
            return []

        doc = docs[0]
        doc_title = file_path.stem

        base_chunks = self.chunker.chunk(doc, config)
        total_chunks = len(base_chunks)
        enriched_chunks: list[Chunk] = []

        for idx, bc in enumerate(base_chunks):
            meta = ChunkMetadata(
                doc_id=doc.id,
                chunk_index=idx,
                total_chunks=total_chunks,
                start_char=bc.metadata.start_char,
                end_char=bc.metadata.end_char,
                token_count=bc.metadata.token_count,
                file_type="text",
                content_type="text",
                section_hierarchy=[doc_title],
                policy_name=doc_title,
                last_modified=mtime,
                source_uri=str(file_path.resolve()),
            )
            contextualized = f"[Document: {doc_title}]\n{bc.text}"
            enriched_chunks.append(
                Chunk(
                    id=bc.id,
                    text=bc.text,
                    contextualized_text=contextualized,
                    metadata=meta,
                )
            )

        return enriched_chunks
