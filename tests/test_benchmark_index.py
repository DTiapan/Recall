"""Tests for persisted benchmark index manifests and reuse."""

from pathlib import Path

from recall.eval.benchmark_index import (
    BenchmarkIndexManifest,
    build_index_fingerprint,
    build_index_slug,
    can_reuse_index,
    is_incomplete_index_slot,
    load_manifest,
    load_subsample_snapshot,
    resolve_benchmark_index,
    save_benchmark_report,
    subsample_snapshot_path,
    wipe_index,
    write_manifest,
    write_subsample_snapshot,
)
from recall.eval.datasets.models import BenchmarkCorpus, BenchmarkDocument, BenchmarkQuery


def test_build_index_slug_is_stable():
    slug = build_index_slug("beir:fiqa", document_limit=500, scale=None, subsample_seed=42, fast=False)
    assert slug == "beir-fiqa_500_s42"


def test_manifest_roundtrip_and_reuse(tmp_path: Path):
    index = resolve_benchmark_index(
        dataset="beir:scifact",
        document_limit=500,
        scale=None,
        subsample_seed=42,
        fast=False,
        index_dir=tmp_path,
    )
    expected = build_index_fingerprint(
        dataset="beir:scifact",
        document_count=500,
        subsample_seed=42,
        scale=None,
        document_limit=500,
        dense_model="BAAI/bge-small-en-v1.5",
        sparse_model="prithivida/Splade_PP_en_v1",
        chunk_size=500,
        rag_mode="local",
        fast_mode=False,
    )
    stored = BenchmarkIndexManifest(
        dataset=expected.dataset,
        document_count=expected.document_count,
        chunk_count=500,
        subsample_seed=expected.subsample_seed,
        scale=expected.scale,
        document_limit=expected.document_limit,
        dense_model=expected.dense_model,
        sparse_model=expected.sparse_model,
        chunk_size=expected.chunk_size,
        rag_mode=expected.rag_mode,
        fast_mode=expected.fast_mode,
    )
    write_manifest(index.manifest_path, stored)

    assert can_reuse_index(index.manifest_path, stored_points=500, expected=expected)
    assert not can_reuse_index(index.manifest_path, stored_points=100, expected=expected)

    mismatch = build_index_fingerprint(
        dataset="beir:fiqa",
        document_count=500,
        subsample_seed=42,
        scale=None,
        document_limit=500,
        dense_model=expected.dense_model,
        sparse_model=expected.sparse_model,
        chunk_size=500,
        rag_mode="local",
        fast_mode=False,
    )
    assert not can_reuse_index(index.manifest_path, stored_points=500, expected=mismatch)
    assert load_manifest(index.manifest_path) is not None


def test_incomplete_index_slot_detection(tmp_path: Path):
    index = resolve_benchmark_index(
        dataset="beir:fiqa",
        document_limit=None,
        scale="10k",
        subsample_seed=42,
        fast=False,
        index_dir=tmp_path,
    )
    assert not is_incomplete_index_slot(index)
    index.qdrant_path.mkdir(parents=True)
    assert is_incomplete_index_slot(index)
    write_manifest(
        index.manifest_path,
        BenchmarkIndexManifest(
            dataset="beir:fiqa",
            document_count=10,
            chunk_count=10,
            subsample_seed=42,
            scale="10k",
            document_limit=10_000,
            dense_model="dense",
            sparse_model="sparse",
            chunk_size=500,
            rag_mode="local",
            fast_mode=False,
        ),
    )
    assert not is_incomplete_index_slot(index)


def test_subsample_snapshot_roundtrip(tmp_path: Path):
    index = resolve_benchmark_index(
        dataset="beir:fiqa",
        document_limit=2,
        scale=None,
        subsample_seed=42,
        fast=False,
        index_dir=tmp_path,
    )
    corpus = BenchmarkCorpus(
        name="beir:fiqa",
        documents=[
            BenchmarkDocument(doc_id="d1", text="a", source_uri="beir:fiqa/d1"),
            BenchmarkDocument(doc_id="d2", text="b", source_uri="beir:fiqa/d2"),
        ],
        queries=[BenchmarkQuery(query_id="q1", query="?", relevant_doc_ids=["d1"])],
        subsample_meta={"seed": 42, "selected_docs": 2},
    )
    path = subsample_snapshot_path(index)
    write_subsample_snapshot(path, corpus)
    loaded = load_subsample_snapshot(path)
    assert loaded is not None
    assert loaded["doc_ids"] == ["d1", "d2"]
    assert loaded["query_count"] == 1


def test_save_benchmark_report_writes_latest_and_archive(tmp_path: Path):
    index = resolve_benchmark_index(
        dataset="beir:fiqa",
        document_limit=500,
        scale=None,
        subsample_seed=42,
        fast=False,
        index_dir=tmp_path,
    )
    stamped, latest = save_benchmark_report(index, "# report\n", rerank=True)
    assert stamped.exists()
    assert latest.exists()
    assert stamped.name.endswith("_rerank.md")
    assert latest.name == "latest_rerank.md"


def test_wipe_index_removes_slot(tmp_path: Path):
    index = resolve_benchmark_index(
        dataset="sample",
        document_limit=None,
        scale=None,
        subsample_seed=42,
        fast=False,
        index_dir=tmp_path,
    )
    index.root_dir.mkdir(parents=True)
    (index.root_dir / "marker.txt").write_text("x", encoding="utf-8")
    wipe_index(index)
    assert not index.root_dir.exists()
