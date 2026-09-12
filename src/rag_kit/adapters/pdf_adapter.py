"""PDF chunking adapter with page-level tracking and table-aware Markdown serialization."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar

from rag_kit.adapters import BaseChunkingAdapter
from rag_kit.core.models import Chunk, ChunkMetadata, IngestConfig
from rag_kit.chunkers.recursive import RecursiveChunker
from rag_kit.chunkers.table_formatter import TableFormatter
from rag_kit.preprocessing.cleaner import clean_text


class PDFChunkingAdapter(BaseChunkingAdapter):
    """Parses PDF documents, extracting text and tables page-by-page.
    
    Preserves 1-indexed page numbers, serializes tables to Markdown,
    and isolates table blocks from standard prose.
    """

    SUPPORTED_EXTENSIONS: ClassVar[set[str]] = {".pdf"}

    def __init__(self, table_formatter: TableFormatter | None = None) -> None:
        self.table_formatter = table_formatter or TableFormatter()
        self.text_chunker = RecursiveChunker()

    def chunk(self, file_path: Path, config: IngestConfig | None = None) -> list[Chunk]:
        import pdfplumber

        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        mtime = datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc)
        doc_id = file_path.stem
        chunks: list[Chunk] = []
        chunk_idx = 0

        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)
            doc_title = (pdf.metadata.get("Title") or file_path.stem).strip()

            for page_idx, page in enumerate(pdf.pages, start=1):
                # 1. Extract and format tables on this page
                tables = page.extract_tables() or []
                table_texts: set[str] = set()

                for tbl_i, tbl in enumerate(tables):
                    if not tbl or len(tbl) < 1:
                        continue
                    tbl_caption = f"{doc_title} (Page {page_idx}, Table {tbl_i + 1})"
                    table_chunks = self.table_formatter.chunk_table(
                        tbl,
                        max_tokens=config.chunk_size if config else 500,
                        caption=tbl_caption,
                    )
                    for tbl_text in table_chunks:
                        token_count = self.table_formatter.count_tokens(tbl_text)
                        meta = ChunkMetadata(
                            doc_id=doc_id,
                            chunk_index=chunk_idx,
                            total_chunks=0,  # Updated at the end
                            start_char=0,
                            end_char=len(tbl_text),
                            token_count=token_count,
                            page_number=page_idx,
                            total_pages=total_pages,
                            file_type="pdf",
                            content_type="table",
                            section_hierarchy=[doc_title, f"Page {page_idx}", f"Table {tbl_i + 1}"],
                            policy_name=doc_title,
                            last_modified=mtime,
                            source_uri=str(file_path.resolve()),
                        )
                        chunk = Chunk(
                            id=f"{doc_id}#p{page_idx}_tbl{tbl_i}_{chunk_idx}",
                            text=tbl_text,
                            contextualized_text=f"[Document: {doc_title}; Page: {page_idx}; Content: Table]\n{tbl_text}",
                            metadata=meta,
                        )
                        chunks.append(chunk)
                        chunk_idx += 1

                # 2. Extract prose text on this page
                raw_text = page.extract_text() or ""
                cleaned = clean_text(raw_text)
                if cleaned:
                    page_doc_id = f"{doc_id}_p{page_idx}"
                    # Create temporary Document container to chunk using RecursiveChunker
                    from rag_kit.core.models import Document
                    temp_doc = Document(id=page_doc_id, content=cleaned, source_uri=str(file_path.resolve()))
                    page_chunks = self.text_chunker.chunk(temp_doc, config)

                    for pc in page_chunks:
                        token_count = pc.metadata.token_count
                        meta = ChunkMetadata(
                            doc_id=doc_id,
                            chunk_index=chunk_idx,
                            total_chunks=0,
                            start_char=pc.metadata.start_char,
                            end_char=pc.metadata.end_char,
                            token_count=token_count,
                            page_number=page_idx,
                            total_pages=total_pages,
                            file_type="pdf",
                            content_type="text",
                            section_hierarchy=[doc_title, f"Page {page_idx}"],
                            policy_name=doc_title,
                            last_modified=mtime,
                            source_uri=str(file_path.resolve()),
                        )
                        chunk = Chunk(
                            id=f"{doc_id}#p{page_idx}_{chunk_idx}",
                            text=pc.text,
                            contextualized_text=f"[Document: {doc_title}; Page: {page_idx}]\n{pc.text}",
                            metadata=meta,
                        )
                        chunks.append(chunk)
                        chunk_idx += 1

        # Update total chunks
        total_chunks = len(chunks)
        for c in chunks:
            c.metadata.total_chunks = total_chunks

        return chunks
