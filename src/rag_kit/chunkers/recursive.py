"""Recursive hierarchical chunker preserving natural structural and sentence boundaries."""

from __future__ import annotations

import tiktoken
from rag_kit.core.interfaces import BaseChunker
from rag_kit.core.models import Chunk, ChunkMetadata, Document, IngestConfig


class RecursiveChunker(BaseChunker):
    """Recursively splits text on natural boundaries (paragraphs -> lines -> sentences -> words)
    while respecting token budget constraints.
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int | None = None,
        tokenizer_name: str = "cl100k_base",
        separators: list[str] | None = None,
    ) -> None:
        self.chunk_size = chunk_size
        if chunk_overlap is None:
            self.chunk_overlap = min(50, max(0, int(chunk_size * 0.1)))
        else:
            if chunk_overlap >= chunk_size:
                raise ValueError("chunk_overlap must be strictly smaller than chunk_size.")
            self.chunk_overlap = chunk_overlap

        self.tokenizer_name = tokenizer_name
        self.separators = separators or self.DEFAULT_SEPARATORS
        self._encoding = tiktoken.get_encoding(tokenizer_name)

    def count_tokens(self, text: str) -> int:
        return len(self._encoding.encode(text, disallowed_special=()))

    def _split_text(self, text: str, separators: list[str], chunk_size: int) -> list[str]:
        final_chunks: list[str] = []
        separator = separators[-1]
        new_separators: list[str] = []

        for i, sep in enumerate(separators):
            if sep == "" or sep in text:
                separator = sep
                new_separators = separators[i + 1 :]
                break

        splits = text.split(separator) if separator != "" else list(text)

        good_splits: list[str] = []
        for s in splits:
            if not s:
                continue
            if self.count_tokens(s) <= chunk_size:
                good_splits.append(s)
            else:
                if new_separators:
                    other_chunks = self._split_text(s, new_separators, chunk_size)
                    good_splits.extend(other_chunks)
                else:
                    good_splits.append(s)

        # Merge adjacent splits up to chunk_size
        current_chunk: list[str] = []
        current_tokens = 0

        for split in good_splits:
            split_tokens = self.count_tokens(split)
            if current_tokens + split_tokens > chunk_size and current_chunk:
                merged = separator.join(current_chunk)
                final_chunks.append(merged)
                current_chunk = []
                current_tokens = 0

            current_chunk.append(split)
            current_tokens += split_tokens

        if current_chunk:
            final_chunks.append(separator.join(current_chunk))

        return final_chunks

    def chunk(self, document: Document, config: IngestConfig | None = None) -> list[Chunk]:
        chunk_size = config.chunk_size if config else self.chunk_size
        text = document.content
        if not text:
            return []

        raw_pieces = self._split_text(text, self.separators, chunk_size)
        total_chunks = len(raw_pieces)
        chunks: list[Chunk] = []

        for idx, piece in enumerate(raw_pieces):
            token_count = self.count_tokens(piece)
            meta = ChunkMetadata(
                doc_id=document.id,
                chunk_index=idx,
                total_chunks=total_chunks,
                start_char=0,
                end_char=len(piece),
                token_count=token_count,
                source_uri=document.source_uri,
            )
            chunks.append(
                Chunk(
                    id=f"{document.id}#{idx}",
                    text=piece,
                    metadata=meta,
                )
            )

        return chunks
