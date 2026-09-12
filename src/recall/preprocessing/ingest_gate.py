"""Document-level deduplication gates for ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import xxhash

from recall.core.models import Document
from recall.preprocessing.cleaner import clean_text
from recall.preprocessing.dedup import ExactDeduplicator, NearDuplicateDetector

_TEXT_SUFFIXES = {".md", ".txt", ".markdown", ".json", ".csv", ".html", ".xml", ".yaml", ".yml"}


@dataclass
class DedupGate:
    exact: ExactDeduplicator
    near: NearDuplicateDetector


class DedupGateRegistry:
    """Per-collection exact and near-duplicate gates."""

    def __init__(self, near_dup_threshold: float = 0.85) -> None:
        self._near_dup_threshold = near_dup_threshold
        self._gates: dict[str, DedupGate] = {}

    def get(self, collection_name: str) -> DedupGate:
        if collection_name not in self._gates:
            self._gates[collection_name] = DedupGate(
                exact=ExactDeduplicator(),
                near=NearDuplicateDetector(threshold=self._near_dup_threshold),
            )
        return self._gates[collection_name]


def document_from_file(file_path: Path) -> Document:
    """Builds a deduplication document from a file path."""
    suffix = file_path.suffix.lower()
    source_uri = str(file_path)

    if suffix in _TEXT_SUFFIXES:
        content = clean_text(file_path.read_text(encoding="utf-8", errors="replace"))
        return Document(content=content, source_uri=source_uri)

    file_bytes = file_path.read_bytes()
    return Document(
        content="",
        source_uri=source_uri,
        content_hash=xxhash.xxh64(file_bytes).hexdigest(),
    )


def should_ingest_document(
    document: Document,
    gate: DedupGate,
    *,
    enable_deduplication: bool,
) -> bool:
    """Returns True when the document is new and should be ingested."""
    if not enable_deduplication:
        return True
    if gate.exact.is_duplicate(document):
        return False
    if document.content and gate.near.is_duplicate(document):
        return False
    gate.exact.register(document)
    if document.content:
        gate.near.register(document)
    return True
