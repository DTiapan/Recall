"""Load bundled real-world files from a local directory."""

from __future__ import annotations

import json
from pathlib import Path

from recall.eval.datasets.models import BenchmarkCorpus, BenchmarkDocument, BenchmarkQuery

SUPPORTED_EXTENSIONS = {".md", ".txt", ".markdown", ".pdf", ".docx"}


def load_local_benchmark(
    root: Path,
    name: str = "sample",
    limit: int | None = None,
) -> BenchmarkCorpus:
    """Loads local documents (MD, TXT, PDF, DOCX) and optional ``eval/queries.jsonl`` ground truth."""
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Benchmark dataset directory not found: {root}")

    documents: list[BenchmarkDocument] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if "eval" in path.parts:
            continue
        doc_id = path.stem
        text = ""
        if path.suffix.lower() in {".md", ".txt", ".markdown"}:
            text = path.read_text(encoding="utf-8")
        documents.append(
            BenchmarkDocument(
                doc_id=doc_id,
                text=text,
                source_uri=str(path.relative_to(root)),
                metadata={"file_name": path.name, "format": path.suffix.lower().lstrip(".")},
            )
        )
        if limit is not None and len(documents) >= limit:
            break

    queries = _load_queries(root / "eval" / "queries.jsonl")
    return BenchmarkCorpus(name=name, documents=documents, queries=queries, data_root=root)


def _load_queries(path: Path) -> list[BenchmarkQuery]:
    if not path.exists():
        return []

    queries: list[BenchmarkQuery] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        relevant_sources = payload.get("relevant_sources", [])
        queries.append(
            BenchmarkQuery(
                query_id=payload["query_id"],
                query=payload["query"],
                relevant_sources=relevant_sources,
                relevant_doc_ids=[Path(src).stem for src in relevant_sources],
                expected_terms=payload.get("expected_terms", []),
            )
        )
    return queries
