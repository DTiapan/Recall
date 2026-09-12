"""Word (.docx) chunking adapter preserving heading hierarchies and tables."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar

from recall.adapters import BaseChunkingAdapter
from recall.core.models import Chunk, ChunkMetadata, IngestConfig
from recall.chunkers.recursive import RecursiveChunker
from recall.chunkers.table_formatter import TableFormatter
from recall.preprocessing.cleaner import clean_text


class DocxChunkingAdapter(BaseChunkingAdapter):
    """Parses Microsoft Word (.docx) documents, maintaining heading hierarchy breadcrumbs
    and converting embedded tables into Markdown.
    """

    SUPPORTED_EXTENSIONS: ClassVar[set[str]] = {".docx"}

    def __init__(self, table_formatter: TableFormatter | None = None) -> None:
        self.table_formatter = table_formatter or TableFormatter()
        self.text_chunker = RecursiveChunker()

    def chunk(self, file_path: Path, config: IngestConfig | None = None) -> list[Chunk]:
        import docx

        if not file_path.exists():
            raise FileNotFoundError(f"Docx file not found: {file_path}")

        mtime = datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc)
        doc = docx.Document(file_path)
        doc_title = file_path.stem
        doc_id = file_path.stem

        chunks: list[Chunk] = []
        chunk_idx = 0
        active_hierarchy: list[str] = [doc_title]

        # Process document elements: paragraphs and tables
        for element in doc.element.body:
            tag = element.tag.split("}")[-1] if "}" in element.tag else element.tag

            if tag == "p":
                # Paragraph
                p = docx.text.paragraph.Paragraph(element, doc)
                text = clean_text(p.text)
                if not text:
                    continue

                style_name = p.style.name if p.style else ""
                if "Heading 1" in style_name:
                    active_hierarchy = [doc_title, text]
                    continue
                elif "Heading 2" in style_name:
                    active_hierarchy = active_hierarchy[:2] + [text]
                    continue
                elif "Heading 3" in style_name:
                    active_hierarchy = active_hierarchy[:3] + [text]
                    continue

                # Normal prose text
                token_count = self.text_chunker.count_tokens(text)
                meta = ChunkMetadata(
                    doc_id=doc_id,
                    chunk_index=chunk_idx,
                    total_chunks=0,
                    start_char=0,
                    end_char=len(text),
                    token_count=token_count,
                    file_type="docx",
                    content_type="text",
                    section_hierarchy=list(active_hierarchy),
                    policy_name=doc_title,
                    last_modified=mtime,
                    source_uri=str(file_path.resolve()),
                )
                breadcrumb_str = " > ".join(active_hierarchy)
                chunks.append(
                    Chunk(
                        id=f"{doc_id}#c{chunk_idx}",
                        text=text,
                        contextualized_text=f"[Document: {breadcrumb_str}]\n{text}",
                        metadata=meta,
                    )
                )
                chunk_idx += 1

            elif tag == "tbl":
                # Table
                t = docx.table.Table(element, doc)
                grid: list[list[str]] = []
                for row in t.rows:
                    grid.append([cell.text.strip() for cell in row.cells])

                if grid and len(grid) >= 1:
                    caption = f"{' > '.join(active_hierarchy)} (Table)"
                    tbl_chunks = self.table_formatter.chunk_table(
                        grid,
                        max_tokens=config.chunk_size if config else 500,
                        caption=caption,
                    )
                    for tbl_text in tbl_chunks:
                        token_count = self.table_formatter.count_tokens(tbl_text)
                        meta = ChunkMetadata(
                            doc_id=doc_id,
                            chunk_index=chunk_idx,
                            total_chunks=0,
                            start_char=0,
                            end_char=len(tbl_text),
                            token_count=token_count,
                            file_type="docx",
                            content_type="table",
                            section_hierarchy=list(active_hierarchy),
                            policy_name=doc_title,
                            last_modified=mtime,
                            source_uri=str(file_path.resolve()),
                        )
                        breadcrumb_str = " > ".join(active_hierarchy)
                        chunks.append(
                            Chunk(
                                id=f"{doc_id}#tbl_{chunk_idx}",
                                text=tbl_text,
                                contextualized_text=f"[Document: {breadcrumb_str}; Content: Table]\n{tbl_text}",
                                metadata=meta,
                            )
                        )
                        chunk_idx += 1

        total_chunks = len(chunks)
        for c in chunks:
            c.metadata.total_chunks = total_chunks

        return chunks
