"""End-to-end retrieval benchmark using real-world corpora and labeled queries."""

from __future__ import annotations

import resource
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from recall.api.service import RAGService
from recall.core.models import Chunk, Document
from recall.preprocessing.ingest_gate import document_from_file
from recall.eval.datasets.beir import load_beir_benchmark
from recall.eval.datasets.local import load_local_benchmark
from recall.eval.datasets.models import BenchmarkCorpus, BenchmarkDocument, BenchmarkQuery


@dataclass
class QueryBenchmarkResult:
    query_id: str
    query: str
    hit_at_5: bool
    reciprocal_rank: float
    latency_ms: float
    hit_at_5_rerank: bool = False
    reciprocal_rank_rerank: float = 0.0
    rerank_latency_ms: float = 0.0
    top_chunk_id: str | None = None


@dataclass
class RetrievalBenchmarkReport:
    dataset_name: str
    documents_ingested: int
    queries_evaluated: int
    hit_rate_at_5: float
    mrr: float
    hit_rate_at_5_rerank: float
    mrr_rerank: float
    latency_p50_ms: float
    latency_p95_ms: float
    rerank_latency_p50_ms: float
    ingest_seconds: float
    ingest_docs_per_sec: float = 0.0
    peak_rss_mb: float = 0.0
    latency_p99_ms: float = 0.0
    query_results: list[QueryBenchmarkResult] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [
            f"# Retrieval Benchmark — {self.dataset_name}",
            "",
            f"- Documents ingested: **{self.documents_ingested}**",
            f"- Queries evaluated: **{self.queries_evaluated}**",
            f"- HitRate@5 (retrieval): **{self.hit_rate_at_5:.1%}**",
            f"- MRR (retrieval): **{self.mrr:.3f}**",
            f"- HitRate@5 (rerank): **{self.hit_rate_at_5_rerank:.1%}**",
            f"- MRR (rerank): **{self.mrr_rerank:.3f}**",
            f"- Ingest time: **{self.ingest_seconds:.2f}s**",
            f"- Ingest throughput: **{self.ingest_docs_per_sec:.1f} docs/sec**",
            f"- Peak RSS: **{self.peak_rss_mb:.0f} MB**",
            f"- Query latency P50: **{self.latency_p50_ms:.1f}ms**",
            f"- Query latency P95: **{self.latency_p95_ms:.1f}ms**",
            f"- Query latency P99: **{self.latency_p99_ms:.1f}ms**",
            f"- Rerank latency P50: **{self.rerank_latency_p50_ms:.1f}ms**",
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
        retrieve_limit: int = 20,
        batch_size: int = 128,
        deduplicate: bool | None = None,
    ) -> RetrievalBenchmarkReport:
        service = self.service or RAGService(default_collection=collection_name)
        use_dedup = deduplicate if deduplicate is not None else len(corpus.documents) < 1000

        ingest_start = time.perf_counter()
        peak_rss_mb = _peak_rss_mb()
        total_docs = len(corpus.documents)
        chunks, ingested = self._prepare_corpus_chunks(
            service=service,
            corpus=corpus,
            collection_name=collection_name,
            deduplicate=use_dedup,
        )

        last_progress_k = 0

        def _on_batch(done: int, total: int) -> None:
            nonlocal peak_rss_mb, last_progress_k
            peak_rss_mb = max(peak_rss_mb, _peak_rss_mb())
            if total < 1000:
                return
            milestone_k = done // 1000
            if milestone_k > last_progress_k:
                last_progress_k = milestone_k
                elapsed = time.perf_counter() - ingest_start
                rate = done / elapsed if elapsed > 0 else 0.0
                print(
                    f"Ingested {done}/{total} chunks ({rate:.1f} chunks/sec)...",
                    flush=True,
                )

        await service.ingest_chunks_batched(
            chunks,
            collection_name=collection_name,
            batch_size=batch_size,
            on_batch=_on_batch,
        )
        ingest_seconds = time.perf_counter() - ingest_start
        ingest_docs_per_sec = ingested / ingest_seconds if ingest_seconds > 0 else 0.0

        query_results: list[QueryBenchmarkResult] = []
        for labeled_query in corpus.queries:
            result = await self._evaluate_query(
                service=service,
                labeled_query=labeled_query,
                collection_name=collection_name,
                search_limit=search_limit,
                retrieve_limit=retrieve_limit,
            )
            query_results.append(result)

        hit_rate = _mean_bool(query_results, "hit_at_5")
        mrr = _mean_attr(query_results, "reciprocal_rank")
        hit_rate_rerank = _mean_bool(query_results, "hit_at_5_rerank")
        mrr_rerank = _mean_attr(query_results, "reciprocal_rank_rerank")
        latencies = [result.latency_ms for result in query_results]
        rerank_latencies = [result.rerank_latency_ms for result in query_results]
        p50 = statistics.median(latencies) if latencies else 0.0
        p95 = _percentile(latencies, 95) if latencies else 0.0
        p99 = _percentile(latencies, 99) if latencies else 0.0
        rerank_p50 = statistics.median(rerank_latencies) if rerank_latencies else 0.0
        peak_rss_mb = max(peak_rss_mb, _peak_rss_mb())

        return RetrievalBenchmarkReport(
            dataset_name=corpus.name,
            documents_ingested=ingested,
            queries_evaluated=len(query_results),
            hit_rate_at_5=hit_rate,
            mrr=mrr,
            hit_rate_at_5_rerank=hit_rate_rerank,
            mrr_rerank=mrr_rerank,
            latency_p50_ms=p50,
            latency_p95_ms=p95,
            latency_p99_ms=p99,
            rerank_latency_p50_ms=rerank_p50,
            ingest_seconds=ingest_seconds,
            ingest_docs_per_sec=ingest_docs_per_sec,
            peak_rss_mb=peak_rss_mb,
            query_results=query_results,
        )

    def _prepare_corpus_chunks(
        self,
        service: RAGService,
        corpus: BenchmarkCorpus,
        collection_name: str,
        deduplicate: bool,
    ) -> tuple[list[Chunk], int]:
        chunks: list[Chunk] = []
        ingested_docs = 0

        for document in corpus.documents:
            file_chunks, accepted = self._chunks_for_document(
                service=service,
                corpus=corpus,
                document=document,
                collection_name=collection_name,
                deduplicate=deduplicate,
            )
            if not accepted:
                continue
            chunks.extend(file_chunks)
            ingested_docs += 1

        return chunks, ingested_docs

    def _chunks_for_document(
        self,
        service: RAGService,
        corpus: BenchmarkCorpus,
        document: BenchmarkDocument,
        collection_name: str,
        deduplicate: bool,
    ) -> tuple[list[Chunk], bool]:
        if corpus.data_root is not None:
            file_path = corpus.data_root / document.source_uri
            if file_path.is_file():
                gate_doc = document_from_file(file_path)
                if deduplicate and not service._should_ingest(gate_doc, collection_name):
                    return [], False
                return service.prepare_file_chunks(file_path), True

        gate_doc = Document(content=document.text, source_uri=document.source_uri)
        if deduplicate and not service._should_ingest(gate_doc, collection_name):
            return [], False

        return [
            service.build_text_chunk(
                text=document.text,
                source_uri=document.source_uri,
                doc_id=document.doc_id,
            )
        ], True

    async def _evaluate_query(
        self,
        service: RAGService,
        labeled_query: BenchmarkQuery,
        collection_name: str,
        search_limit: int,
        retrieve_limit: int,
    ) -> QueryBenchmarkResult:
        start = time.perf_counter()
        results = await service.search(
            query=labeled_query.query,
            limit=search_limit,
            collection_name=collection_name,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0

        rerank_start = time.perf_counter()
        reranked = await service.search_rerank(
            query=labeled_query.query,
            limit=search_limit,
            collection_name=collection_name,
            retrieve_limit=retrieve_limit,
        )
        rerank_latency_ms = (time.perf_counter() - rerank_start) * 1000.0

        top_chunk_id = results[0].chunk_id if results else None
        return QueryBenchmarkResult(
            query_id=labeled_query.query_id,
            query=labeled_query.query,
            hit_at_5=_is_hit(results, labeled_query, search_limit),
            reciprocal_rank=_reciprocal_rank(results, labeled_query),
            latency_ms=latency_ms,
            hit_at_5_rerank=_is_hit(reranked, labeled_query, search_limit),
            reciprocal_rank_rerank=_reciprocal_rank(reranked, labeled_query),
            rerank_latency_ms=rerank_latency_ms,
            top_chunk_id=top_chunk_id,
        )


def _mean_bool(results: list[QueryBenchmarkResult], field_name: str) -> float:
    if not results:
        return 0.0
    return sum(1 for result in results if getattr(result, field_name)) / len(results)


def _mean_attr(results: list[QueryBenchmarkResult], field_name: str) -> float:
    if not results:
        return 0.0
    return statistics.mean(getattr(result, field_name) for result in results)


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


def _peak_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage / (1024 * 1024)
    return usage / 1024


def resolve_dataset(
    dataset: str,
    dataset_path: Path | None,
    limit: int | None,
    scale: str | None,
    query_limit: int | None = None,
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
        corpus = load_beir_benchmark(
            beir_name,
            limit=effective_limit,
            query_limit=query_limit,
        )
        if scale:
            corpus.name = f"{corpus.name}@{scale}"
        return corpus

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
