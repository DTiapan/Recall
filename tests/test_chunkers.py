"""Tests for chunking strategies: FixedToken, Recursive, and Contextual."""

import pytest
from recall.core.models import Document, IngestConfig
from recall.chunkers.fixed_token import FixedTokenChunker
from recall.chunkers.recursive import RecursiveChunker
from recall.chunkers.contextual import ContextualChunker


def test_fixed_token_chunker_basic():
    text = "The quick brown fox jumps over the lazy dog. " * 50
    doc = Document(id="doc-fixed", content=text)

    chunker = FixedTokenChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk(doc)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.metadata.token_count <= 100
        assert chunk.metadata.doc_id == "doc-fixed"
        assert chunk.id.startswith("doc-fixed#")


def test_recursive_chunker_preserves_paragraphs():
    p1 = "Paragraph one discusses Q1 results in detail with financial metrics."
    p2 = "Paragraph two details engineering milestones achieved during the period."
    p3 = "Paragraph three outlines risks and compliance considerations."
    text = f"{p1}\n\n{p2}\n\n{p3}"
    doc = Document(id="doc-rec", content=text)

    # Each paragraph has ~10 tokens, total is 30 tokens. Setting chunk_size=12 forces split per paragraph.
    chunker = RecursiveChunker(chunk_size=12)
    chunks = chunker.chunk(doc)

    assert len(chunks) == 3
    assert chunks[0].text == p1
    assert chunks[1].text == p2
    assert chunks[2].text == p3


def test_contextual_chunker_injects_metadata_breadcrumbs():
    doc = Document(
        id="sec-10q",
        content="Revenue grew by 14% to $500M during the third fiscal quarter.",
        metadata={
            "title": "Acme Corp Q3 2023 Form 10-Q",
            "headings": ["Financial Highlights", "Consolidated Results"],
        },
    )

    chunker = ContextualChunker()
    chunks = chunker.chunk(doc)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.text == "Revenue grew by 14% to $500M during the third fiscal quarter."
    assert "[Document: Acme Corp Q3 2023 Form 10-Q; Sections: Financial Highlights > Consolidated Results]" in chunk.contextualized_text
    assert chunk.searchable_text.startswith("[Document: Acme Corp")


def test_chunker_empty_document():
    doc = Document(id="empty", content="")
    fixed = FixedTokenChunker().chunk(doc)
    recursive = RecursiveChunker().chunk(doc)
    contextual = ContextualChunker().chunk(doc)

    assert fixed == []
    assert recursive == []
    assert contextual == []
