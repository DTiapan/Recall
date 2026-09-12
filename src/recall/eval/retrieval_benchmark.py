"""End-to-end retrieval benchmark using real-world corpora and labeled queries."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

from recall.api.service import RAGService
from recall.eval.datasets.beir import load_beir_benchmark
from recall.eval.datasets.local import load_local_benchmark
from recall.eval.datasets.models import BenchmarkCorpus, BenchmarkQuery


@dataclass
class QueryBenchmarkResult:
    query_id: str
    query: str
    hit_at_5: bool
    reciprocal_rank: float
    latency_ms: float
    top_chunk_id: str | None = None


@dataclass
class RetrievalBenchmarkReport:
    dataset_name: str
    documents_ingested: int
    queries_evaluated: int
    hit_rate_at_5: float
    mrr: float
    latency_p50_ms: float
    latency_p95_ms: float
    ingest_seconds: float
    query_results: list[QueryBenchmarkResult] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [
            f"# Retrieval Benchmark — {self.dataset_name}",
            "",
            f"- Documents ingested: **{self.documents_ingested}**",
            f"- Queries evaluated: **{self.queries_evaluated}**",
            f"- HitRate@5: **{self.hit_rate_at_5:.1%}**",
            f"- MRR: **{self.mrr:.3f}**",
            f"- Ingest time: **{self.ingest_seconds:.2f}s**",
            f"- Query latency P50: **{self.latency_p50_ms:.1f}ms**",
            f"- Query latency P95: **{self.latency_p95_ms:.1f}ms**",
        ]
        return "\n".join(lines)


class RetrievalBenchmarkRunner:
    """Runs ingest + hybrid search evaluation against a real labeled corpus."""

    def __init__(self, service: RAGService | None = None) -> None:
        self.service = service

    async def run(
        self,
        corpus: BenchmarkCorpus,
        collection_name: str = "benchmark",
        search_limit: int = 5,
    ) -> RetrievalBenchmarkReport:
        service = self.service or RAGService(default_collection=collection_name)

        ingest_start = time.perf_counter()
        ingested = 0
        for document in corpus.documents:
            count = await service.ingest_text(
                text=document.text,
                source_uri=document.source_uri,
                collection_name=collection_name,
                doc_id=document.doc_id,
            )
            if count > 0:
                ingested += 1
        ingest_seconds = time.perf_counter() - ingest_start

        query_results: list[QueryBenchmarkResult] = []
        for labeled_query in corpus.queries:
            result = await self._evaluate_query(
                service=service,
                labeled_query=labeled_query,
                collection_name=collection_name,
                search_limit=search_limit,
            )
            query_results.append(result)

        hit_rate = (
            sum(1 for result in query_results if result.hit_at_5) / len(query_results)
            if query_results
            else 0.0
        )
        mrr = (
            statistics.mean(result.reciprocal_rank for result in query_results)
            if query_results
            else 0.0
        )
        latencies = [result.latency_ms for result in query_results]
        p50 = statistics.median(latencies) if latencies else 0.0
        p95 = _percentile(latencies, 95) if latencies else 0.0

        return RetrievalBenchmarkReport(
            dataset_name=corpus.name,
            documents_ingested=ingested,
            queries_evaluated=len(query_results),
            hit_rate_at_5=hit_rate,
            mrr=mrr,
            latency_p50_ms=p50,
            latency_p95_ms=p95,
            ingest_seconds=ingest_seconds,
            query_results=query_results,
        )

    async def _evaluate_query(
        self,
        service: RAGService,
        labeled_query: BenchmarkQuery,
        collection_name: str,
        search_limit: int,
    ) -> QueryBenchmarkResult:
        start = time.perf_counter()
        results = await service.search(
            query=labeled_query.query,
            limit=search_limit,
            collection_name=collection_name,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0

        hit_at_5 = _is_hit(results, labeled_query, search_limit)
        reciprocal_rank = _reciprocal_rank(results, labeled_query)

        top_chunk_id = results[0].chunk_id if results else None
        return QueryBenchmarkResult(
            query_id=labeled_query.query_id,
            query=labeled_query.query,
            hit_at_5=hit_at_5,
            reciprocal_rank=reciprocal_rank,
            latency_ms=latency_ms,
            top_chunk_id=top_chunk_id,
        )


def _is_hit(results, labeled_query: BenchmarkQuery, limit: int) -> bool:
    if not results:
        return False

    relevant_doc_ids = set(labeled_query.relevant_doc_ids)
    relevant_sources = set(labeled_query.relevant_sources)

    for result in results[:limit]:
        metadata = result.metadata
        if metadata.doc_id in relevant_doc_ids:
            return True
        source_uri = metadata.source_uri or ""
        if any(source in source_uri for source in relevant_sources):
            return True
        if labeled_query.expected_terms and any(
            term.lower() in result.text.lower() for term in labeled_query.expected_terms
        ):
            return True
    return False


def _reciprocal_rank(results, labeled_query: BenchmarkQuery) -> float:
    for rank, result in enumerate(results, start=1):
        if _is_hit([result], labeled_query, 1):
            return 1.0 / rank
    return 0.0


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((percentile / 100) * (len(ordered) - 1)))
    return ordered[index]


def resolve_dataset(
    dataset: str,
    dataset_path: Path | None,
    limit: int | None,
    scale: str | None,
) -> BenchmarkCorpus:
    """Resolves a dataset identifier to a real-world ``BenchmarkCorpus``."""
    effective_limit = _scale_to_limit(scale) if scale else limit

    if dataset_path is not None:
        return load_local_benchmark(dataset_path, name=dataset_path.name, limit=effective_limit)

    if dataset == "sample":
        root = Path(__file__).resolve().parents[3] / "data" / "sample"
        return load_local_benchmark(root, name="sample", limit=effective_limit)

    if dataset.startswith("beir:"):
        beir_name = dataset.split(":", 1)[1]
        return load_beir_benchmark(beir_name, limit=effective_limit)

    raise ValueError(
        f"Unknown dataset '{dataset}'. Use 'sample', 'beir:<name>', or --dataset-path."
    )


def _scale_to_limit(scale: str) -> int:
    mapping = {
        "10k": 10_000,
        "100k": 100_000,
        "1m": 1_000_000,
        "10m": 10_000_000,
    }
    normalized = scale.lower().strip()
    if normalized not in mapping:
        raise ValueError(f"Unsupported scale '{scale}'. Use one of: 10k, 100k, 1m, 10m")
    return mapping[normalized]
