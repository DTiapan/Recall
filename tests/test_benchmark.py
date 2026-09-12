"""Tests for ChunkingBenchmarkRunner and report generation."""

import pytest
from rag_kit.core.models import Document
from rag_kit.eval.benchmark import ChunkingBenchmarkRunner, BenchmarkReport


def test_benchmark_runner_evaluation():
    sample_text = (
        "Enterprise architectures require strict performance SLAs. "
        "Each service must maintain independent database connections.\n\n"
        "Vector indexes at 10M scale require Scalar Quantization (INT8)."
    )
    docs = [
        Document(id=f"doc_{i}", content=sample_text, metadata={"title": f"Doc {i}", "headings": ["SLA", "Quantization"]})
        for i in range(3)
    ]

    runner = ChunkingBenchmarkRunner()
    report = runner.evaluate(docs)

    assert isinstance(report, BenchmarkReport)
    assert report.total_documents == 3
    assert report.total_input_tokens > 0
    assert len(report.metrics) == 3

    for m in report.metrics:
        assert m.total_chunks > 0
        assert m.mean_tokens > 0
        assert m.throughput_tokens_sec > 0

    md_table = report.to_markdown_table()
    assert "| Strategy |" in md_table
    assert "FixedToken" in md_table
    assert "Recursive" in md_table
    assert "Contextual" in md_table


def test_benchmark_runner_empty_input():
    runner = ChunkingBenchmarkRunner()
    report = runner.evaluate([])
    assert report.total_documents == 0
    assert report.total_input_tokens == 0
    assert report.metrics == []
