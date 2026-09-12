"""Data models for retrieval benchmark corpora and queries."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BenchmarkDocument:
    doc_id: str
    text: str
    source_uri: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class BenchmarkQuery:
    query_id: str
    query: str
    relevant_doc_ids: list[str] = field(default_factory=list)
    relevant_sources: list[str] = field(default_factory=list)
    expected_terms: list[str] = field(default_factory=list)


@dataclass
class BenchmarkCorpus:
    name: str
    documents: list[BenchmarkDocument]
    queries: list[BenchmarkQuery]
    data_root: Path | None = None
