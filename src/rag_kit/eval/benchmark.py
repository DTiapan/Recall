"""Comparative benchmark harness for evaluating chunking strategies across performance and quality metrics."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence
import numpy as np
from rich.console import Console
from rich.table import Table

from rag_kit.core.interfaces import BaseChunker
from rag_kit.core.models import Chunk, Document
from rag_kit.chunkers.fixed_token import FixedTokenChunker
from rag_kit.chunkers.recursive import RecursiveChunker
from rag_kit.chunkers.contextual import ContextualChunker


@dataclass
class StrategyMetrics:
    name: str
    total_chunks: int
    mean_tokens: float
    min_tokens: int
    max_tokens: int
    std_tokens: float
    boundary_integrity_pct: float
    elapsed_ms: float
    throughput_tokens_sec: float


@dataclass
class BenchmarkReport:
    total_documents: int
    total_input_tokens: int
    metrics: list[StrategyMetrics]

    def to_markdown_table(self) -> str:
        headers = [
            "Strategy",
            "Chunks",
            "Mean Tokens",
            "Token Range",
            "Token StdDev",
            "Boundary Integrity",
            "Latency (ms)",
            "Throughput (tokens/s)",
        ]
        lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        for m in self.metrics:
            row = [
                m.name,
                str(m.total_chunks),
                f"{m.mean_tokens:.1f}",
                f"{m.min_tokens} - {m.max_tokens}",
                f"{m.std_tokens:.1f}",
                f"{m.boundary_integrity_pct:.1f}%",
                f"{m.elapsed_ms:.2f}ms",
                f"{m.throughput_tokens_sec:,.0f}",
            ]
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)


class ChunkingBenchmarkRunner:
    """Runs a battery of documents through multiple chunking strategies and computes
    statistical, structural, and throughput benchmarks.
    """

    def __init__(self, chunkers: dict[str, BaseChunker] | None = None) -> None:
        self.chunkers = chunkers or {
            "FixedToken (500/50)": FixedTokenChunker(chunk_size=500, chunk_overlap=50),
            "Recursive (500/50)": RecursiveChunker(chunk_size=500, chunk_overlap=50),
            "Contextual (Recursive+Meta)": ContextualChunker(base_chunker=RecursiveChunker(chunk_size=500, chunk_overlap=50)),
        }

    @staticmethod
    def _ends_on_boundary(text: str) -> bool:
        stripped = text.strip()
        return stripped.endswith((".", "!", "?", '."', '!"', '?"', "```", ":"))

    def evaluate(self, documents: Sequence[Document]) -> BenchmarkReport:
        if not documents:
            return BenchmarkReport(total_documents=0, total_input_tokens=0, metrics=[])

        # Reference tokenizer for baseline token counts
        ref_chunker = FixedTokenChunker()
        total_input_tokens = sum(ref_chunker.count_tokens(doc.content) for doc in documents)

        metrics_list: list[StrategyMetrics] = []

        for name, chunker in self.chunkers.items():
            start_time = time.perf_counter()
            all_chunks: list[Chunk] = []

            for doc in documents:
                chunks = chunker.chunk(doc)
                all_chunks.extend(chunks)

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            throughput = (total_input_tokens / (elapsed_ms / 1000.0)) if elapsed_ms > 0 else 0.0

            if not all_chunks:
                metrics_list.append(
                    StrategyMetrics(
                        name=name,
                        total_chunks=0,
                        mean_tokens=0.0,
                        min_tokens=0,
                        max_tokens=0,
                        std_tokens=0.0,
                        boundary_integrity_pct=0.0,
                        elapsed_ms=elapsed_ms,
                        throughput_tokens_sec=0.0,
                    )
                )
                continue

            token_counts = np.array([c.metadata.token_count for c in all_chunks])
            boundary_clean = sum(1 for c in all_chunks if self._ends_on_boundary(c.text))
            boundary_pct = (boundary_clean / len(all_chunks)) * 100.0

            metrics_list.append(
                StrategyMetrics(
                    name=name,
                    total_chunks=len(all_chunks),
                    mean_tokens=float(np.mean(token_counts)),
                    min_tokens=int(np.min(token_counts)),
                    max_tokens=int(np.max(token_counts)),
                    std_tokens=float(np.std(token_counts)),
                    boundary_integrity_pct=boundary_pct,
                    elapsed_ms=elapsed_ms,
                    throughput_tokens_sec=throughput,
                )
            )

        return BenchmarkReport(
            total_documents=len(documents),
            total_input_tokens=total_input_tokens,
            metrics=metrics_list,
        )

    def print_report(self, report: BenchmarkReport) -> None:
        console = Console()
        table = Table(title=f"Chunking Strategy Benchmark ({report.total_documents} Docs, {report.total_input_tokens} Tokens)")
        table.add_column("Strategy", style="cyan bold")
        table.add_column("Chunks", justify="right")
        table.add_column("Mean Tokens", justify="right")
        table.add_column("Token Range", justify="right")
        table.add_column("Token StdDev", justify="right")
        table.add_column("Boundary Integrity", justify="right", style="green")
        table.add_column("Latency (ms)", justify="right")
        table.add_column("Throughput (tok/s)", justify="right", style="magenta")

        for m in report.metrics:
            table.add_row(
                m.name,
                str(m.total_chunks),
                f"{m.mean_tokens:.1f}",
                f"{m.min_tokens} - {m.max_tokens}",
                f"{m.std_tokens:.1f}",
                f"{m.boundary_integrity_pct:.1f}%",
                f"{m.elapsed_ms:.2f}ms",
                f"{m.throughput_tokens_sec:,.0f}",
            )

        console.print(table)


if __name__ == "__main__":
    # Self-contained sample benchmark
    sample_text = (
        "# Enterprise Cloud Infrastructure Architecture\n\n"
        "Modern cloud-native systems require resilient designs spanning multiple availability zones. "
        "Each service must maintain independent database connections with circuit breakers to prevent cascading failures.\n\n"
        "## Performance Metrics and Latency Budgets\n\n"
        "The P99 retrieval latency is bounded at 150 milliseconds. "
        "Requests exceeding this threshold automatically degrade to sparse index lookups, ensuring 99.99% availability.\n\n"
        "### Scalability and Quantization\n\n"
        "Vector databases handling 10 million documents require Scalar Quantization (INT8) to fit in RAM. "
        "This achieves a 75% memory footprint reduction while retaining over 98% recall.\n"
    ) * 10

    docs = [
        Document(
            id=f"benchmark_doc_{i}",
            content=sample_text,
            metadata={"title": f"Infrastructure Spec Volume {i}", "headings": ["Architecture", "Performance", "Scalability"]},
        )
        for i in range(5)
    ]

    runner = ChunkingBenchmarkRunner()
    rep = runner.evaluate(docs)
    runner.print_report(rep)
