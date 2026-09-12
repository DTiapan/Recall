"""Lexical BM25 sparse index for exact match, acronym, and technical code retrieval."""

from __future__ import annotations

import re
from typing import Any
from rank_bm25 import BM25Plus

from recall.core.interfaces import BaseSparseIndex
from recall.core.models import Chunk, SearchResult


def tokenize_lexical(text: str) -> list[str]:
    """Tokenizes text preserving technical alphanumeric identifiers, dashes, and codes
    e.g. 'CVE-2024-3094', 'RFC-4122', '401(k)', 'TS-902'.
    """
    cleaned = text.lower()
    # Matches words with hyphens and alphanumeric sequences
    tokens = re.findall(r"\b[a-z0-9]+(?:[-_][a-z0-9]+)*\b", cleaned)
    return tokens if tokens else cleaned.split()


class BM25Index:
    """In-memory BM25+ lexical index supporting exact keyword search
    and metadata payload filtering. Uses BM25+ to prevent negative IDF
    artifacts and ensure lower-bounded relevance scoring.
    """

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._bm25: BM25Plus | None = None
        self._tokenized_corpus: list[list[str]] = []

    def count(self) -> int:
        return len(self._chunks)

    def index(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0

        self._chunks = list(chunks)
        self._tokenized_corpus = [tokenize_lexical(c.searchable_text) for c in self._chunks]
        self._bm25 = BM25Plus(self._tokenized_corpus)
        return len(self._chunks)

    def search(
        self,
        query: str,
        limit: int = 10,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        if not self._bm25 or not self._chunks:
            return []

        tokenized_query = tokenize_lexical(query)
        if not tokenized_query:
            return []

        scores = self._bm25.get_scores(tokenized_query)

        # Pair scores with chunk objects
        scored_candidates: list[tuple[float, Chunk]] = []
        for score, chunk in zip(scores, self._chunks):
            if score <= 0.0:
                continue

            # Apply metadata filters if provided
            if filter_dict:
                match = True
                meta_dict = chunk.metadata.model_dump()
                for k, v in filter_dict.items():
                    if meta_dict.get(k) != v and chunk.metadata.extra.get(k) != v:
                        match = False
                        break
                if not match:
                    continue

            scored_candidates.append((float(score), chunk))

        # Sort descending by BM25 score
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        top_candidates = scored_candidates[:limit]

        results: list[SearchResult] = []
        for score, chunk in top_candidates:
            results.append(
                SearchResult(
                    chunk_id=chunk.id,
                    score=score,
                    text=chunk.text,
                    metadata=chunk.metadata,
                    vector_name="sparse",
                )
            )

        return results
