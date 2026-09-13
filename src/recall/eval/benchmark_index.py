"""Persistent on-disk benchmark indexes with manifest-based reuse."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from recall.eval.datasets.models import BenchmarkCorpus


COLLECTION_NAME = "benchmark"


@dataclass(frozen=True)
class BenchmarkIndexContext:
    """Filesystem layout for a reproducible benchmark index slot."""

    slug: str
    root_dir: Path
    qdrant_path: Path
    manifest_path: Path
    collection_name: str = COLLECTION_NAME


@dataclass(frozen=True)
class BenchmarkIndexManifest:
    """Fingerprint of an indexed benchmark corpus — must match to skip re-ingest."""

    dataset: str
    document_count: int
    chunk_count: int
    subsample_seed: int
    scale: str | None
    document_limit: int | None
    dense_model: str
    sparse_model: str
    chunk_size: int
    rag_mode: str
    fast_mode: bool

    def to_dict(self) -> dict[str, int | str | bool | None]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> BenchmarkIndexManifest:
        return cls(
            dataset=str(data["dataset"]),
            document_count=int(data["document_count"]),
            chunk_count=int(data["chunk_count"]),
            subsample_seed=int(data["subsample_seed"]),
            scale=str(data["scale"]) if data.get("scale") is not None else None,
            document_limit=int(data["document_limit"]) if data.get("document_limit") is not None else None,
            dense_model=str(data["dense_model"]),
            sparse_model=str(data["sparse_model"]),
            chunk_size=int(data["chunk_size"]),
            rag_mode=str(data["rag_mode"]),
            fast_mode=bool(data.get("fast_mode", False)),
        )


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip().lower())
    return cleaned.strip("-") or "dataset"


def build_index_slug(
    dataset: str,
    document_limit: int | None,
    scale: str | None,
    subsample_seed: int,
    fast: bool,
) -> str:
    parts = [_slugify(dataset.replace(":", "-"))]
    if scale:
        parts.append(scale)
    elif document_limit is not None:
        parts.append(str(document_limit))
    parts.append(f"s{subsample_seed}")
    if fast:
        parts.append("fast")
    return "_".join(parts)


def resolve_benchmark_index(
    dataset: str,
    document_limit: int | None,
    scale: str | None,
    subsample_seed: int,
    fast: bool,
    index_dir: Path,
) -> BenchmarkIndexContext:
    slug = build_index_slug(dataset, document_limit, scale, subsample_seed, fast)
    root_dir = index_dir / slug
    return BenchmarkIndexContext(
        slug=slug,
        root_dir=root_dir,
        qdrant_path=root_dir / "qdrant",
        manifest_path=root_dir / "manifest.json",
    )


def load_manifest(path: Path) -> BenchmarkIndexManifest | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return BenchmarkIndexManifest.from_dict(data)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def write_manifest(path: Path, manifest: BenchmarkIndexManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fingerprint_matches(expected: BenchmarkIndexManifest, stored: BenchmarkIndexManifest) -> bool:
    """Compare index identity fields (chunk_count verified separately via point count)."""
    return (
        expected.dataset == stored.dataset
        and expected.document_count == stored.document_count
        and expected.subsample_seed == stored.subsample_seed
        and expected.scale == stored.scale
        and expected.document_limit == stored.document_limit
        and expected.dense_model == stored.dense_model
        and expected.sparse_model == stored.sparse_model
        and expected.chunk_size == stored.chunk_size
        and expected.rag_mode == stored.rag_mode
        and expected.fast_mode == stored.fast_mode
    )


def can_reuse_index(
    manifest_path: Path,
    stored_points: int,
    expected: BenchmarkIndexManifest,
) -> bool:
    stored = load_manifest(manifest_path)
    if stored is None:
        return False
    if not fingerprint_matches(expected, stored):
        return False
    return stored_points >= stored.chunk_count


def build_index_fingerprint(
    dataset: str,
    document_count: int,
    subsample_seed: int,
    scale: str | None,
    document_limit: int | None,
    dense_model: str,
    sparse_model: str,
    chunk_size: int,
    rag_mode: str,
    fast_mode: bool,
) -> BenchmarkIndexManifest:
    """Build a pre-ingest fingerprint (chunk_count filled in after ingest)."""
    return BenchmarkIndexManifest(
        dataset=dataset,
        document_count=document_count,
        chunk_count=0,
        subsample_seed=subsample_seed,
        scale=scale,
        document_limit=document_limit,
        dense_model=dense_model,
        sparse_model=sparse_model,
        chunk_size=chunk_size,
        rag_mode=rag_mode,
        fast_mode=fast_mode,
    )


def wipe_index(index: BenchmarkIndexContext) -> None:
    if index.root_dir.exists():
        shutil.rmtree(index.root_dir)


def subsample_snapshot_path(index: BenchmarkIndexContext) -> Path:
    return index.root_dir / "subsample.json"


def reports_dir(index: BenchmarkIndexContext) -> Path:
    return index.root_dir / "reports"


def is_incomplete_index_slot(index: BenchmarkIndexContext) -> bool:
    """True when Qdrant data exists on disk but the manifest fingerprint is missing."""
    return index.qdrant_path.exists() and not index.manifest_path.is_file()


def write_subsample_snapshot(path: Path, corpus: BenchmarkCorpus) -> None:
    """Persist the selected BEIR doc IDs and subsample metadata for reproducibility."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": corpus.name,
        "doc_ids": [doc.doc_id for doc in corpus.documents],
        "query_count": len(corpus.queries),
        "subsample_meta": corpus.subsample_meta or {},
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_subsample_snapshot(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, TypeError):
        return None


def save_benchmark_report(
    index: BenchmarkIndexContext,
    markdown: str,
    *,
    rerank: bool = False,
) -> tuple[Path, Path]:
    """Write timestamped and latest benchmark reports under the index slot."""
    report_dir = reports_dir(index)
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = "rerank" if rerank else "hybrid"
    stamped = report_dir / f"{stamp}_{suffix}.md"
    latest = report_dir / f"latest_{suffix}.md"
    stamped.write_text(markdown, encoding="utf-8")
    latest.write_text(markdown, encoding="utf-8")
    return stamped, latest
