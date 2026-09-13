"""CI retrieval regression gate tests."""

from __future__ import annotations

import pytest

from recall.eval.retrieval_gate import (
    RetrievalGateError,
    RetrievalGateFloors,
    SAMPLE_CORPUS_FLOORS,
    check_report_meets_floors,
)
from recall.eval.retrieval_benchmark import RetrievalBenchmarkReport


def test_gate_passes_at_baseline():
    report = RetrievalBenchmarkReport(
        dataset_name="sample",
        documents_ingested=5,
        queries_evaluated=10,
        hit_rate_at_5=1.0,
        mrr=0.662,
        hit_rate_at_5_rerank=0.0,
        mrr_rerank=0.0,
        latency_p50_ms=1.0,
        latency_p95_ms=1.0,
        rerank_latency_p50_ms=0.0,
        ingest_seconds=1.0,
    )
    check_report_meets_floors(report, SAMPLE_CORPUS_FLOORS)


def test_gate_fails_on_hit_rate_regression():
    report = RetrievalBenchmarkReport(
        dataset_name="sample",
        documents_ingested=5,
        queries_evaluated=10,
        hit_rate_at_5=0.5,
        mrr=0.662,
        hit_rate_at_5_rerank=0.0,
        mrr_rerank=0.0,
        latency_p50_ms=1.0,
        latency_p95_ms=1.0,
        rerank_latency_p50_ms=0.0,
        ingest_seconds=1.0,
    )
    with pytest.raises(RetrievalGateError, match="HitRate@5"):
        check_report_meets_floors(report, SAMPLE_CORPUS_FLOORS)


def test_gate_fails_on_mrr_regression():
    report = RetrievalBenchmarkReport(
        dataset_name="sample",
        documents_ingested=5,
        queries_evaluated=10,
        hit_rate_at_5=1.0,
        mrr=0.1,
        hit_rate_at_5_rerank=0.0,
        mrr_rerank=0.0,
        latency_p50_ms=1.0,
        latency_p95_ms=1.0,
        rerank_latency_p50_ms=0.0,
        ingest_seconds=1.0,
    )
    with pytest.raises(RetrievalGateError, match="MRR"):
        check_report_meets_floors(report, SAMPLE_CORPUS_FLOORS)


def test_custom_floors_partial():
    floors = RetrievalGateFloors(ndcg_at_10=0.5)
    report = RetrievalBenchmarkReport(
        dataset_name="beir:fiqa",
        documents_ingested=100,
        queries_evaluated=50,
        hit_rate_at_5=0.3,
        mrr=0.2,
        hit_rate_at_5_rerank=0.0,
        mrr_rerank=0.0,
        latency_p50_ms=1.0,
        latency_p95_ms=1.0,
        rerank_latency_p50_ms=0.0,
        ingest_seconds=1.0,
        ndcg_at_10=0.52,
    )
    check_report_meets_floors(report, floors)
