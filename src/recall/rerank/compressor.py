"""Extractive context compressor to eliminate prompt noise and prevent Lost-in-the-Middle."""

from __future__ import annotations

import re
import tiktoken
from recall.core.interfaces import BaseContextCompressor
from recall.core.models import SearchResult


class ContextCompressor:
    """Extracts high-salience sentences from retrieved chunks based on query alignment,
    discarding boilerplate and reducing LLM context window token costs.
    """

    def __init__(self, tokenizer_name: str = "cl100k_base") -> None:
        try:
            self._tokenizer = tiktoken.get_encoding(tokenizer_name)
        except Exception:
            self._tokenizer = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text, disallowed_special=()))

    def _split_sentences(self, text: str) -> list[str]:
        """Splits text into sentences while preserving structural integrity."""
        # Split on paragraph boundaries or sentence end punctuation
        raw_sentences = re.split(r"(?<=[.!?])\s+|\n\n+", text)
        return [s.strip() for s in raw_sentences if s.strip()]

    def compress_chunk(
        self,
        query: str,
        text: str,
        max_tokens: int = 200,
    ) -> str:
        """Compresses a single chunk text to salient sentences within max_tokens."""
        if max_tokens <= 0:
            return ""

        total_tokens = self._count_tokens(text)
        if total_tokens <= max_tokens:
            return text

        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            # Cannot split further, truncate safely by token length
            tokens = self._tokenizer.encode(text, disallowed_special=())[:max_tokens]
            return self._tokenizer.decode(tokens)

        query_terms = set(re.findall(r"\b\w+\b", query.lower()))

        # Score sentences: query term overlap + position bonus
        scored_sentences: list[tuple[int, float, str, int]] = []
        for idx, sentence in enumerate(sentences):
            sent_tokens = self._count_tokens(sentence)
            words = set(re.findall(r"\b\w+\b", sentence.lower()))
            overlap = len(query_terms.intersection(words))

            # Saliency score: keyword matches + slight preference for early topic sentences
            score = overlap * 2.0 + (1.0 / (idx + 1))
            scored_sentences.append((idx, score, sentence, sent_tokens))

        # Sort descending by saliency score
        scored_sentences.sort(key=lambda x: x[1], reverse=True)

        selected_indices: list[int] = []
        accumulated_tokens = 0

        for original_idx, score, sentence, sent_tokens in scored_sentences:
            if accumulated_tokens + sent_tokens > max_tokens:
                if not selected_indices and sent_tokens > max_tokens:
                    # Single sentence exceeds max_tokens; truncate to budget
                    truncated_tokens = self._tokenizer.encode(sentence, disallowed_special=())[:max_tokens]
                    sentences[original_idx] = self._tokenizer.decode(truncated_tokens)
                    selected_indices.append(original_idx)
                    accumulated_tokens += max_tokens
                    break
                continue

            selected_indices.append(original_idx)
            accumulated_tokens += sent_tokens
            if accumulated_tokens >= max_tokens:
                break

        if not selected_indices:
            tokens = self._tokenizer.encode(text, disallowed_special=())[:max_tokens]
            return self._tokenizer.decode(tokens)

        # Re-order selected sentences into original chronological sequence
        selected_indices.sort()
        selected_text = " [...] ".join(sentences[i] for i in selected_indices)
        return selected_text

    def compress(
        self,
        query: str,
        candidates: list[SearchResult],
        max_tokens_per_chunk: int = 200,
        max_total_tokens: int = 1500,
    ) -> list[SearchResult]:
        """Compresses a list of candidate search results while enforcing token budgets."""
        compressed_results: list[SearchResult] = []
        running_total_tokens = 0

        for candidate in candidates:
            remaining_budget = max_total_tokens - running_total_tokens
            if remaining_budget <= 5:  # Skip chunks if remaining budget is negligible
                break

            chunk_budget = min(max_tokens_per_chunk, remaining_budget)

            compressed_text = self.compress_chunk(
                query=query,
                text=candidate.text,
                max_tokens=chunk_budget,
            )
            compressed_tokens = self._count_tokens(compressed_text)
            if not compressed_text or compressed_tokens == 0:
                continue

            running_total_tokens += compressed_tokens

            compressed_cand = candidate.model_copy(
                update={"text": compressed_text}
            )
            compressed_results.append(compressed_cand)

        return compressed_results
