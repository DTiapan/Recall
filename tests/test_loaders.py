"""Tests for multi-format document loaders (Text, Markdown, JSON)."""

import json
from pathlib import Path
import pytest
from recall.loaders.text import TextLoader
from recall.loaders.markdown import MarkdownLoader
from recall.loaders.json_loader import JSONLoader


def test_text_loader(tmp_path: Path):
    file = tmp_path / "sample.txt"
    file.write_text("First line of text.\nSecond line of text.", encoding="utf-8")

    loader = TextLoader()
    docs = loader.load(str(file))
    assert len(docs) == 1
    assert "First line of text." in docs[0].content
    assert docs[0].metadata["filename"] == "sample.txt"


def test_markdown_loader_with_frontmatter_and_headings(tmp_path: Path):
    file = tmp_path / "doc.md"
    content = (
        "---\n"
        "title: Q3 Financial Overview\n"
        "author: Finance Team\n"
        "---\n"
        "# Executive Summary\n"
        "The quarter showed robust performance.\n\n"
        "## Operating Metrics\n"
        "Operating margins reached 24%.\n"
    )
    file.write_text(content, encoding="utf-8")

    loader = MarkdownLoader()
    docs = loader.load(str(file))
    assert len(docs) == 1
    doc = docs[0]
    assert doc.metadata["title"] == "Q3 Financial Overview"
    assert doc.metadata["author"] == "Finance Team"
    assert "Executive Summary" in doc.metadata["headings"]
    assert "Operating Metrics" in doc.metadata["headings"]
    assert "The quarter showed robust performance." in doc.content


def test_json_loader(tmp_path: Path):
    file = tmp_path / "records.json"
    data = [
        {"id": "rec-1", "text": "Customer inquiry regarding refund policy.", "category": "billing"},
        {"id": "rec-2", "text": "Technical issue with API authentication token.", "category": "tech"},
    ]
    file.write_text(json.dumps(data), encoding="utf-8")

    loader = JSONLoader(text_key="text", id_key="id")
    docs = loader.load(str(file))
    assert len(docs) == 2
    assert docs[0].id == "rec-1"
    assert docs[0].metadata["category"] == "billing"
    assert docs[1].id == "rec-2"
    assert docs[1].metadata["category"] == "tech"
