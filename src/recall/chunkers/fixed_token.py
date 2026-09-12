"""Fixed-size token window chunker with configurable token overlap."""

from __future__ import annotations

import tiktoken
from recall.core.interfaces import BaseChunker
from recall.core.models import Chunk, ChunkMetadata, Document, IngestConfig


class FixedTokenChunker(BaseChunker):
    """Chunks documents into strict token windows using tiktoken encoding."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int | None = None,
        tokenizer_name: str = "cl100k_base",
    ) -> None:
        self.chunk_size = chunk_size
        if chunk_overlap is None:
            self.chunk_overlap = min(50, max(0, int(chunk_size * 0.1)))
        else:
            if chunk_overlap >= chunk_size:
                raise ValueError("chunk_overlap must be strictly smaller than chunk_size.")
            self.chunk_overlap = chunk_overlap

        self.tokenizer_name = tokenizer_name
        self._encoding = tiktoken.get_encoding(tokenizer_name)

    def count_tokens(self, text: str) -> int:
        return len(self._encoding.encode(text, disallowed_special=()))

    def chunk(self, document: Document, config: IngestConfig | None = None) -> list[Chunk]:
        chunk_size = config.chunk_size if config else self.chunk_size
        chunk_overlap = config.chunk_overlap if config else self.chunk_overlap

        text = document.content
        if not text:
            return []

        tokens = self._encoding.encode(text, disallowed_special=())
        total_tokens = len(tokens)

        if total_tokens <= chunk_size:
            meta = ChunkMetadata(
                doc_id=document.id,
                chunk_index=0,
                total_chunks=1,
                start_char=0,
                end_char=len(text),
                token_count=total_tokens,
                source_uri=document.source_uri,
            )
            return [Chunk(id=f"{document.id}#0", text=text, metadata=meta)]

        step = max(1, chunk_size - chunk_overlap)
        chunks: list[Chunk] = []
        token_offsets: list[int] = list(range(0, total_tokens, step))
        total_count = len(token_offsets)

        for idx, start_token in enumerate(token_offsets):
            end_token = min(start_token + chunk_size, total_tokens)
            chunk_tokens = tokens[start_token:end_token]
            chunk_text = self._encoding.decode(chunk_tokens)

            meta = ChunkMetadata(
                doc_id=document.id,
                chunk_index=idx,
                total_chunks=total_count,
                start_char=0,
                end_char=len(chunk_text),
                token_count=len(chunk_tokens),
                source_uri=document.source_uri,
            )
            chunks.append(
                Chunk(
                    id=f"{document.id}#{idx}",
                    text=chunk_text,
                    metadata=meta,
                )
            )

        return chunks
