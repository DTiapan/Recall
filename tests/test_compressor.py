"""Unit tests for extractive ContextCompressor."""

import pytest
from recall.core.interfaces import BaseContextCompressor
from recall.core.models import ChunkMetadata, SearchResult
from recall.rerank import ContextCompressor


def _make_candidate(chunk_id: str, text: str) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        score=0.9,
        text=text,
        metadata=ChunkMetadata(doc_id=f"doc_{chunk_id}", chunk_index=0, source_uri=f"file_{chunk_id}.md"),
        vector_name="fused",
    )


def test_compressor_protocol_conformance():
    compressor = ContextCompressor()
    assert isinstance(compressor, BaseContextCompressor)


def test_compressor_short_text_intact():
    compressor = ContextCompressor()
    short_text = "This is a short statement about Python."
    candidates = [_make_candidate("c1", short_text)]

    results = compressor.compress(
        query="Python",
        candidates=candidates,
        max_tokens_per_chunk=100,
    )

    assert len(results) == 1
    assert results[0].text == short_text
    assert results[0].metadata.source_uri == "file_c1.md"


def test_compressor_extracts_salient_sentences():
    compressor = ContextCompressor()
    long_text = (
        "Welcome to the company handbook introduction. "
        "The company was founded in nineteen ninety nine by two college graduates. "
        "To request remote work approval, employees must submit form HR-402 in Workday. "
        "The kitchen is stocked with snacks and espresso every Monday morning. "
        "Remote work requests are processed within three business days by the manager. "
        "Please ensure your desk is kept tidy at the end of the day."
    )
    candidates = [_make_candidate("c1", long_text)]

    # Compress targeting remote work with a tight token budget
    results = compressor.compress(
        query="How to request remote work approval?",
        candidates=candidates,
        max_tokens_per_chunk=35,
    )

    assert len(results) == 1
    compressed_text = results[0].text
    # Should contain the remote work sentences and exclude the snack/desk boilerplate
    assert "remote work approval" in compressed_text.lower() or "hr-402" in compressed_text.lower()
    assert "snacks and espresso" not in compressed_text


def test_compressor_total_budget_limit():
    compressor = ContextCompressor()
    c1 = _make_candidate("c1", "Sentence one with ten words for candidate one here today. " * 3)
    c2 = _make_candidate("c2", "Sentence two with ten words for candidate two here today. " * 3)
    c3 = _make_candidate("c3", "Sentence three with ten words for candidate three here today. " * 3)

    # Set very small max_total_tokens to cap how many candidates can be accommodated
    results = compressor.compress(
        query="candidate",
        candidates=[c1, c2, c3],
        max_tokens_per_chunk=20,
        max_total_tokens=25,
    )

    assert len(results) <= 2
