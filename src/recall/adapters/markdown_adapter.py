"""Markdown chunking adapter extracting frontmatter, heading hierarchies, and isolated tables."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar

from recall.adapters import BaseChunkingAdapter
from recall.chunkers.recursive import RecursiveChunker
from recall.chunkers.table_formatter import TableFormatter
from recall.core.models import Chunk, ChunkMetadata, Document, IngestConfig
from recall.loaders.markdown import MarkdownLoader
from recall.preprocessing.cleaner import clean_text


def _extract_blocks(markdown_text: str) -> list[tuple[str, str]]:
    """Segments markdown text into alternating blocks of ('table', table_text)
    and ('text', prose_text).
    """
    lines = markdown_text.splitlines()
    blocks: list[tuple[str, str]] = []
    current_prose: list[str] = []
    current_table: list[str] = []

    def flush_prose() -> None:
        if current_prose:
            content = "\n".join(current_prose).strip()
            if content:
                blocks.append(("text", content))
            current_prose.clear()

    def flush_table() -> None:
        if current_table:
            # Must have at least 2 lines and a separator line to be a valid markdown table
            has_sep = any("---" in line for line in current_table)
            content = "\n".join(current_table).strip()
            if has_sep and len(current_table) >= 2:
                flush_prose()
                blocks.append(("table", content))
            else:
                # Not a real table, treat as prose
                current_prose.extend(current_table)
            current_table.clear()

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            current_table.append(line)
        else:
            if current_table:
                flush_table()
            current_prose.append(line)

    if current_table:
        flush_table()
    flush_prose()

    return blocks


class MarkdownChunkingAdapter(BaseChunkingAdapter):
    """Parses Markdown documents, maintaining active heading hierarchy breadcrumbs,
    extracting frontmatter policy metadata, and isolating Markdown tables.
    """

    SUPPORTED_EXTENSIONS: ClassVar[set[str]] = {".md", ".markdown"}

    def __init__(self) -> None:
        self.loader = MarkdownLoader(parse_frontmatter=True)
        self.chunker = RecursiveChunker()
        self.table_formatter = TableFormatter()

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
        headings = doc.metadata.get("headings", [])

        blocks = _extract_blocks(doc.content)
        chunks: list[Chunk] = []
        chunk_idx = 0

        for b_type, b_content in blocks:
            if b_type == "table":
                token_count = self.chunker.count_tokens(b_content)
                hierarchy = [doc_title]
                if headings:
                    hierarchy.extend(headings[:2])

                meta = ChunkMetadata(
                    doc_id=doc.id,
                    chunk_index=chunk_idx,
                    token_count=token_count,
                    file_type="markdown",
                    content_type="table",
                    section_hierarchy=hierarchy,
                    policy_name=policy_name,
                    last_modified=mtime,
                    source_uri=str(file_path.resolve()),
                    extra=doc.metadata,
                )
                breadcrumb_str = " > ".join(hierarchy)
                contextualized = f"[Document: {breadcrumb_str}; Type: table]\n{b_content}"

                chunks.append(
                    Chunk(
                        id=f"{doc.id}#{chunk_idx}",
                        text=b_content,
                        contextualized_text=contextualized,
                        metadata=meta,
                    )
                )
                chunk_idx += 1
            else:
                # Prose text block
                sub_doc = Document(id=doc.id, content=b_content, metadata=doc.metadata)
                sub_chunks = self.chunker.chunk(sub_doc, config)
                for sc in sub_chunks:
                    hierarchy = [doc_title]
                    if headings:
                        hierarchy.extend(headings[:2])

                    meta = ChunkMetadata(
                        doc_id=doc.id,
                        chunk_index=chunk_idx,
                        start_char=sc.metadata.start_char,
                        end_char=sc.metadata.end_char,
                        token_count=sc.metadata.token_count,
                        file_type="markdown",
                        content_type="text",
                        section_hierarchy=hierarchy,
                        policy_name=policy_name,
                        last_modified=mtime,
                        source_uri=str(file_path.resolve()),
                        extra=doc.metadata,
                    )
                    breadcrumb_str = " > ".join(hierarchy)
                    contextualized = f"[Document: {breadcrumb_str}; Type: text]\n{sc.text}"

                    chunks.append(
                        Chunk(
                            id=f"{doc.id}#{chunk_idx}",
                            text=sc.text,
                            contextualized_text=contextualized,
                            metadata=meta,
                        )
                    )
                    chunk_idx += 1

        # Backfill total_chunks
        for c in chunks:
            c.metadata.total_chunks = len(chunks)

        return chunks
