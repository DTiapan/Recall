"""End-to-end retrieval benchmark using real-world corpora and labeled queries."""

from __future__ import annotations

import resource
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from recall.api.service import RAGService
from recall.core.models import Chunk, Document
from recall.preprocessing.ingest_gate import document_from_file
from recall.eval.datasets.beir import load_beir_benchmark
from recall.eval.datasets.local import load_local_benchmark
from recall.eval.datasets.models import BenchmarkCorpus, BenchmarkDocument, BenchmarkQuery
from recall.eval.benchmark_index import (
    BenchmarkIndexContext,
    BenchmarkIndexManifest,
    can_reuse_index,
    load_manifest,
    subsample_snapshot_path,
    write_manifest,
    write_subsample_snapshot,
)
from recall.eval.trec_metrics import compute_beir_metrics

BEIR_EVAL_TOP_K = 100


@dataclass
class QueryBenchmarkResult:
    query_id: str
    query: str
    hit_at_5: bool
    reciprocal_rank: float
    latency_ms: float
    ranked_doc_scores: dict[str, float] = field(default_factory=dict)
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
    ndcg_at_10: float | None = None
    recall_at_10: float | None = None
    recall_at_100: float | None = None
    mrr_at_10: float | None = None
    subsample_meta: dict[str, int | str] | None = None
    index_reused: bool = False
    index_path: str | None = None
    rerank_evaluated: bool = False
    rerank_candidate_k: int | None = None
    query_results: list[QueryBenchmarkResult] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [
            f"# Retrieval Benchmark — {self.dataset_name}",
            "",
            f"- Documents ingested: **{self.documents_ingested}**",
            f"- Queries evaluated: **{self.queries_evaluated}**",
            f"- HitRate@5 (hybrid retrieval): **{self.hit_rate_at_5:.1%}**",
            f"- MRR (hybrid retrieval): **{self.mrr:.3f}**",
        ]
        if self.ndcg_at_10 is not None:
            lines.extend(
                [
                    f"- nDCG@10 (pytrec_eval): **{self.ndcg_at_10:.3f}**",
                    f"- Recall@10: **{self.recall_at_10:.3f}**",
                    f"- Recall@100: **{self.recall_at_100:.3f}**",
                    f"- MRR@10 (recip_rank): **{self.mrr_at_10:.3f}**",
                ]
            )
        if self.subsample_meta:
            lines.append(
                f"- Subsample: **{self.subsample_meta.get('selected_docs')} docs**, "
                f"seed **{self.subsample_meta.get('seed')}**, "
                f"**{self.subsample_meta.get('evaluable_queries')}** evaluable queries in corpus"
            )
        if self.index_path:
            reused = "reused" if self.index_reused else "built"
            lines.append(f"- Persisted index ({reused}): **{self.index_path}**")
        if self.rerank_evaluated:
            pool = self.rerank_candidate_k if self.rerank_candidate_k is not None else "?"
            lines.extend(
                [
                    f"- HitRate@5 (rerank, pool={pool}): **{self.hit_rate_at_5_rerank:.1%}**",
                    f"- MRR (rerank): **{self.mrr_rerank:.3f}**",
                    f"- Rerank latency P50: **{self.rerank_latency_p50_ms:.1f}ms**",
                ]
            )
        lines.extend(
            [
                f"- Ingest time: **{self.ingest_seconds:.2f}s**",
                f"- Ingest throughput: **{self.ingest_docs_per_sec:.1f} docs/sec**",
                f"- Peak RSS: **{self.peak_rss_mb:.0f} MB**",
                f"- Query latency P50: **{self.latency_p50_ms:.1f}ms**",
                f"- Query latency P95: **{self.latency_p95_ms:.1f}ms**",
                f"- Query latency P99: **{self.latency_p99_ms:.1f}ms**",
            ]
        )
        return "\n".join(lines)


def _log_progress(message: str) -> None:
    print(message, flush=True)


class RetrievalBenchmarkRunner:
    """Runs ingest + hybrid search evaluation against a real labeled corpus."""

    def __init__(self, service: RAGService | None = None) -> None:
        self.service = service

    async def run(
        self,
        corpus: BenchmarkCorpus,
        collection_name: str = "benchmark",
        search_limit: int = 5,
        eval_top_k: int = BEIR_EVAL_TOP_K,
        retrieve_limit: int | None = None,
        batch_size: int = 128,
        deduplicate: bool | None = None,
        include_rerank: bool = False,
        index_context: BenchmarkIndexContext | None = None,
        index_fingerprint: BenchmarkIndexManifest | None = None,
        reuse_index: bool = True,
    ) -> RetrievalBenchmarkReport:
        service = self.service or RAGService(default_collection=collection_name)
        effective_retrieve_limit = retrieve_limit or service.config.pipeline.reranking.candidate_k
        use_dedup = deduplicate if deduplicate is not None else len(corpus.documents) < 1000

        ingest_start = time.perf_counter()
        peak_rss_mb = _peak_rss_mb()
        index_reused = False
        ingested = 0
        total_chunks = 0

        if (
            index_context is not None
            and index_fingerprint is not None
            and reuse_index
            and service.vector_store.collection_exists(collection_name)
        ):
            stored_points = service.sparse_chunk_count(collection_name)
            if can_reuse_index(index_context.manifest_path, stored_points, index_fingerprint):
                index_reused = True
                stored_manifest = load_manifest(index_context.manifest_path)
                ingested = stored_manifest.document_count if stored_manifest else len(corpus.documents)
                total_chunks = stored_manifest.chunk_count if stored_manifest else stored_points
                _log_progress(
                    f"Reusing persisted index at {index_context.root_dir} "
                    f"({stored_points} points, skipped ingest)"
                )

        if not index_reused:
            total_docs = len(corpus.documents)
            _log_progress(
                f"Phase 1/3 — streaming ingest of {total_docs} documents "
                f"(dedup={'on' if use_dedup else 'off'}, batch_size={batch_size})..."
            )

            def _on_batch(done: int, total: int) -> None:
                nonlocal peak_rss_mb
                peak_rss_mb = max(peak_rss_mb, _peak_rss_mb())
                elapsed = time.perf_counter() - ingest_start
                rate = done / elapsed if elapsed > 0 else 0.0
                pct = (100.0 * done / total) if total else 0.0
                remaining = (total - done) / rate if rate > 0 else 0.0
                _log_progress(
                    f"Phase 2/3 — ingest {done}/{total} chunks "
                    f"({pct:.1f}%, {rate:.1f} chunks/sec, ETA {remaining:.0f}s)..."
                )

            ingested, total_chunks = await self._streaming_ingest(
                service=service,
                corpus=corpus,
                collection_name=collection_name,
                deduplicate=use_dedup,
                batch_size=batch_size,
                on_batch=_on_batch,
            )
            _log_progress(
                f"Phase 2/3 complete — {ingested} docs → {total_chunks} chunks "
                f"in {time.perf_counter() - ingest_start:.1f}s"
            )

            if index_context is not None and index_fingerprint is not None:
                manifest = BenchmarkIndexManifest(
                    dataset=index_fingerprint.dataset,
                    document_count=ingested,
                    chunk_count=total_chunks,
                    subsample_seed=index_fingerprint.subsample_seed,
                    scale=index_fingerprint.scale,
                    document_limit=index_fingerprint.document_limit,
                    dense_model=index_fingerprint.dense_model,
                    sparse_model=index_fingerprint.sparse_model,
                    chunk_size=index_fingerprint.chunk_size,
                    rag_mode=index_fingerprint.rag_mode,
                    fast_mode=index_fingerprint.fast_mode,
                )
                write_manifest(index_context.manifest_path, manifest)
                write_subsample_snapshot(subsample_snapshot_path(index_context), corpus)
                _log_progress(
                    f"Persisted index fingerprint → {index_context.manifest_path} "
                    f"({total_chunks} vectors on disk)"
                )

        ingest_seconds = time.perf_counter() - ingest_start
        ingest_docs_per_sec = ingested / ingest_seconds if ingest_seconds > 0 and not index_reused else 0.0
        if not index_reused:
            _log_progress(
                f"Ingest throughput: {ingest_docs_per_sec:.1f} docs/sec, "
                f"peak RSS {_peak_rss_mb():.0f} MB"
            )

        query_results: list[QueryBenchmarkResult] = []
        total_queries = len(corpus.queries)
        rerank_note = (
            f", rerank pool={effective_retrieve_limit}"
            if include_rerank
            else ""
        )
        _log_progress(
            f"Phase 3/3 — evaluating {total_queries} queries "
            f"(hybrid top-{eval_top_k} for BEIR metrics{rerank_note})..."
        )
        for index, labeled_query in enumerate(corpus.queries, start=1):
            result = await self._evaluate_query(
                service=service,
                labeled_query=labeled_query,
                collection_name=collection_name,
                search_limit=search_limit,
                eval_top_k=eval_top_k,
                retrieve_limit=effective_retrieve_limit,
                include_rerank=include_rerank,
            )
            query_results.append(result)
            if index == 1 or index % 5 == 0 or index == total_queries:
                _log_progress(
                    f"  Query {index}/{total_queries} ({labeled_query.query_id}) — "
                    f"hit@5={'yes' if result.hit_at_5 else 'no'}, "
                    f"{result.latency_ms:.0f}ms"
                )

        hit_rate = _mean_bool(query_results, "hit_at_5")
        mrr = _mean_attr(query_results, "reciprocal_rank")
        latencies = [result.latency_ms for result in query_results]
        p50 = statistics.median(latencies) if latencies else 0.0
        p95 = _percentile(latencies, 95) if latencies else 0.0
        p99 = _percentile(latencies, 99) if latencies else 0.0
        peak_rss_mb = max(peak_rss_mb, _peak_rss_mb())

        beir_metrics: dict[str, float] = {}
        if corpus.qrels:
            run = {
                result.query_id: result.ranked_doc_scores
                for result in query_results
                if result.ranked_doc_scores
            }
            beir_metrics = compute_beir_metrics(
                corpus.qrels,
                run,
                k_values=[10, 100],
            )

        rerank_latencies = [result.rerank_latency_ms for result in query_results if include_rerank]
        rerank_p50 = statistics.median(rerank_latencies) if rerank_latencies else 0.0

        return RetrievalBenchmarkReport(
            dataset_name=corpus.name,
            documents_ingested=ingested,
            queries_evaluated=len(query_results),
            hit_rate_at_5=hit_rate,
            mrr=mrr,
            hit_rate_at_5_rerank=_mean_bool(query_results, "hit_at_5_rerank")
            if include_rerank
            else 0.0,
            mrr_rerank=_mean_attr(query_results, "reciprocal_rank_rerank") if include_rerank else 0.0,
            latency_p50_ms=p50,
            latency_p95_ms=p95,
            latency_p99_ms=p99,
            rerank_latency_p50_ms=rerank_p50,
            rerank_evaluated=include_rerank,
            rerank_candidate_k=effective_retrieve_limit if include_rerank else None,
            ingest_seconds=ingest_seconds,
            ingest_docs_per_sec=ingest_docs_per_sec,
            peak_rss_mb=peak_rss_mb,
            ndcg_at_10=beir_metrics.get("ndcg@10"),
            recall_at_10=beir_metrics.get("recall@10"),
            recall_at_100=beir_metrics.get("recall@100"),
            mrr_at_10=beir_metrics.get("mrr@10"),
            subsample_meta=corpus.subsample_meta,
            index_reused=index_reused,
            index_path=str(index_context.root_dir) if index_context else None,
            query_results=query_results,
        )

    async def _streaming_ingest(
        self,
        service: RAGService,
        corpus: BenchmarkCorpus,
        collection_name: str,
        deduplicate: bool,
        batch_size: int,
        on_batch: Callable[[int, int], None],
    ) -> tuple[int, int]:
        """Chunk, embed, and upsert in micro-batches without holding the full corpus in RAM."""
        buffer: list = []
        ingested_docs = 0
        total_chunks = 0
        total_docs = len(corpus.documents)
        progress_step = max(1, total_docs // 20)

        for index, document in enumerate(corpus.documents, start=1):
            file_chunks, accepted = self._chunks_for_document(
                service=service,
                corpus=corpus,
                document=document,
                collection_name=collection_name,
                deduplicate=deduplicate,
            )
            if not accepted:
                continue
            ingested_docs += 1
            buffer.extend(file_chunks)

            while len(buffer) >= batch_size:
                batch = buffer[:batch_size]
                buffer = buffer[batch_size:]
                await service.ingest_chunks_batched(batch, collection_name=collection_name, batch_size=batch_size)
                total_chunks += len(batch)
                on_batch(total_chunks, total_chunks + len(buffer))

            if index == 1 or index % progress_step == 0 or index == total_docs:
                _log_progress(
                    f"  Chunked {index}/{total_docs} documents "
                    f"({ingested_docs} accepted, {total_chunks + len(buffer)} chunks buffered)..."
                )

        if buffer:
            await service.ingest_chunks_batched(buffer, collection_name=collection_name, batch_size=batch_size)
            total_chunks += len(buffer)
            on_batch(total_chunks, total_chunks)

        return ingested_docs, total_chunks

    def _prepare_corpus_chunks(
        self,
        service: RAGService,
        corpus: BenchmarkCorpus,
        collection_name: str,
        deduplicate: bool,
    ) -> tuple[list[Chunk], int]:
        chunks: list[Chunk] = []
        ingested_docs = 0
        total_docs = len(corpus.documents)
        progress_step = max(1, total_docs // 20)

        for index, document in enumerate(corpus.documents, start=1):
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
            if index == 1 or index % progress_step == 0 or index == total_docs:
                _log_progress(
                    f"  Chunked {index}/{total_docs} documents "
                    f"({ingested_docs} accepted, {len(chunks)} chunks)..."
                )

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
        eval_top_k: int,
        retrieve_limit: int,
        include_rerank: bool,
    ) -> QueryBenchmarkResult:
        start = time.perf_counter()
        results = await service.search(
            query=labeled_query.query,
            limit=eval_top_k,
            collection_name=collection_name,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0

        ranked_doc_scores = _results_to_doc_run(results)
        top_chunk_id = results[0].chunk_id if results else None

        hit_at_5_rerank = False
        reciprocal_rank_rerank = 0.0
        rerank_latency_ms = 0.0
        if include_rerank:
            rerank_start = time.perf_counter()
            reranked = await service.search_rerank(
                query=labeled_query.query,
                limit=search_limit,
                collection_name=collection_name,
                retrieve_limit=retrieve_limit,
            )
            rerank_latency_ms = (time.perf_counter() - rerank_start) * 1000.0
            hit_at_5_rerank = _is_hit(reranked, labeled_query, search_limit)
            reciprocal_rank_rerank = _reciprocal_rank(reranked, labeled_query)

        return QueryBenchmarkResult(
            query_id=labeled_query.query_id,
            query=labeled_query.query,
            hit_at_5=_is_hit(results, labeled_query, search_limit),
            reciprocal_rank=_reciprocal_rank(results, labeled_query),
            latency_ms=latency_ms,
            ranked_doc_scores=ranked_doc_scores,
            hit_at_5_rerank=hit_at_5_rerank,
            reciprocal_rank_rerank=reciprocal_rank_rerank,
            rerank_latency_ms=rerank_latency_ms,
            top_chunk_id=top_chunk_id,
        )


def _results_to_doc_run(results) -> dict[str, float]:
    """Map hybrid results to BEIR run format (doc_id -> score), best score per doc."""
    run: dict[str, float] = {}
    for result in results:
        doc_id = result.metadata.doc_id
        if not doc_id:
            continue
        run[doc_id] = max(run.get(doc_id, 0.0), float(result.score))
    return run


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
    subsample_seed: int = 42,
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
            subsample_seed=subsample_seed,
        )
        if scale:
            corpus.name = f"{corpus.name}@{scale}"
        if corpus.subsample_meta:
            meta = corpus.subsample_meta
            _log_progress(
                f"Subsample: {meta.get('selected_docs')} docs (seed={meta.get('seed')}), "
                f"{meta.get('evaluable_queries')} evaluable queries"
            )
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
