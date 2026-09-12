"""Markdown chunking adapter extracting frontmatter and structural heading hierarchies."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar

from rag_kit.adapters import BaseChunkingAdapter
from rag_kit.core.models import Chunk, ChunkMetadata, IngestConfig
from rag_kit.chunkers.recursive import RecursiveChunker
from rag_kit.loaders.markdown import MarkdownLoader
from rag_kit.preprocessing.cleaner import clean_text


class MarkdownChunkingAdapter(BaseChunkingAdapter):
    """Parses Markdown documents, maintaining active heading hierarchy breadcrumbs
    and extracting frontmatter policy metadata.
    """

    SUPPORTED_EXTENSIONS: ClassVar[set[str]] = {".md", ".markdown"}

    def __init__(self) -> None:
        self.loader = MarkdownLoader(parse_frontmatter=True)
        self.chunker = RecursiveChunker()

    def chunk(self, file_path: Path, config: IngestConfig | None = None) -> list[Chunk]:
        if not file_path.exists():
            raise FileNotFoundError(f"Markdown file not found: {file_path}")

        mtime = datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc)
        docs = self.loader.load(str(file_path))
        if not docs:
            return []

        doc = docs[0]
        doc_title = doc.metadata.get("title") or file_path.stem
        policy_name = doc.metadata.get("policy") or doc.metadata.get("policy_name") or doc_title

        base_chunks = self.chunker.chunk(doc, config)
        total_chunks = len(base_chunks)

        enriched_chunks: list[Chunk] = []
        headings = doc.metadata.get("headings", [])

        for idx, bc in enumerate(base_chunks):
            # Infer content type (table or code or text)
            content_type = "text"
            if bc.text.strip().startswith("|") and "---" in bc.text:
                content_type = "table"
            elif "```" in bc.text:
                content_type = "code"

            hierarchy = [doc_title]
            if headings:
                hierarchy.extend(headings[:2])

            meta = ChunkMetadata(
                doc_id=doc.id,
                chunk_index=idx,
                total_chunks=total_chunks,
                start_char=bc.metadata.start_char,
                end_char=bc.metadata.end_char,
                token_count=bc.metadata.token_count,
                file_type="markdown",
                content_type=content_type,
                section_hierarchy=hierarchy,
                policy_name=policy_name,
                last_modified=mtime,
                source_uri=str(file_path.resolve()),
                extra=doc.metadata,
            )

            breadcrumb_str = " > ".join(hierarchy)
            contextualized = f"[Document: {breadcrumb_str}; Type: {content_type}]\n{bc.text}"

            enriched_chunks.append(
                Chunk(
                    id=bc.id,
                    text=bc.text,
                    contextualized_text=contextualized,
                    metadata=meta,
                )
            )

        return enriched_chunks
