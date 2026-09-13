"""Command-line interface (CLI) for Recall."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys
import uvicorn

from recall.api.service import RAGService
from recall.core.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="recall",
        description="Recall: The Open-Source, Turnkey Enterprise RAG Platform",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: recall serve
    serve_parser = subparsers.add_parser("serve", help="Start the FastAPI REST server and Web UI")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind (default: 0.0.0.0)")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")

    # Command: recall ingest
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a document into the RAG vector index")
    ingest_parser.add_argument("file", type=str, help="Path to document file (PDF, DOCX, Markdown, TXT, JSON)")
    ingest_parser.add_argument("--collection", default="documents", help="Target collection name (default: documents)")

    # Command: recall query
    query_parser = subparsers.add_parser("query", help="Query the RAG pipeline directly from the terminal")
    query_parser.add_argument("question", type=str, help="User query question")
    query_parser.add_argument("--collection", default="documents", help="Target collection name (default: documents)")

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Run retrieval benchmarks on real-world datasets (sample or BEIR)",
    )
    benchmark_parser.add_argument(
        "--dataset",
        default="sample",
        help="Dataset id: sample, beir:scifact, beir:fiqa, ... (default: sample)",
    )
    benchmark_parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Optional path to a local real-document corpus directory",
    )
    benchmark_parser.add_argument(
        "--scale",
        choices=["10k", "100k", "1m", "10m"],
        default=None,
        help="Subsample large real corpora to a target document count",
    )
    benchmark_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of real documents to ingest",
    )
    benchmark_parser.add_argument(
        "--collection",
        default="benchmark",
        help="Target Qdrant collection for benchmark ingest (default: benchmark)",
    )
    benchmark_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write markdown benchmark report",
    )
    benchmark_parser.add_argument(
        "--query-limit",
        type=int,
        default=None,
        help="Maximum labeled queries to evaluate (default: all evaluable queries)",
    )
    benchmark_parser.add_argument(
        "--subsample-seed",
        type=int,
        default=42,
        help="RNG seed for BEIR document subsampling (default: 42)",
    )
    benchmark_parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Embedding and Qdrant upsert batch size (default: 128)",
    )
    benchmark_parser.add_argument(
        "--fast",
        action="store_true",
        help="High-throughput scale mode: mock embeddings for 100k–10M index/latency stress tests",
    )
    benchmark_parser.add_argument(
        "--index-dir",
        type=str,
        default=None,
        help="Persist Qdrant index on disk for reuse (default: ~/.cache/recall/benchmark-indexes)",
    )
    benchmark_parser.add_argument(
        "--in-memory",
        action="store_true",
        help="Use ephemeral in-memory Qdrant (discard index after run; no reuse)",
    )
    benchmark_parser.add_argument(
        "--no-reuse-index",
        action="store_true",
        help="Always re-embed and re-ingest even if a matching persisted index exists",
    )
    benchmark_parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Delete persisted index for this dataset slot and rebuild from scratch",
    )
    benchmark_parser.add_argument(
        "--rerank",
        action="store_true",
        help="Also evaluate cross-encoder rerank path (uses reranking.candidate_k from config)",
    )
    benchmark_parser.add_argument(
        "--rerank-pool",
        type=int,
        default=None,
        help="Override hybrid candidate pool size for rerank eval (default: config reranking.candidate_k)",
    )

    args = parser.parse_args()

    if not args.command or args.command == "serve":
        host = getattr(args, "host", "0.0.0.0")
        port = getattr(args, "port", 8000)
        reload = getattr(args, "reload", False)
        print(f"Starting Recall Enterprise RAG on http://{host}:{port}...")
        uvicorn.run("recall.api.app:create_app", host=host, port=port, factory=True, reload=reload)

    elif args.command == "ingest":
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Error: File not found: {file_path}", file=sys.stderr)
            sys.exit(1)

        async def _run_ingest():
            service = RAGService()
            count = await service.ingest_file(file_path, collection_name=args.collection)
            print(f"Successfully indexed {count} chunks from {file_path.name} into collection '{args.collection}'.")

        asyncio.run(_run_ingest())

    elif args.command == "query":
        async def _run_query():
            service = RAGService()
            response = await service.query(args.question, collection_name=args.collection)
            print("\n--- Answer ---")
            print(response.answer)
            if response.citations:
                print("\n--- Citations ---")
                for c in response.citations:
                    page_str = f", p. {c.page_number}" if c.page_number else ""
                    print(f"[{c.doc_index}] {c.source_uri}{page_str}: \"{c.snippet[:80]}...\"")

        asyncio.run(_run_query())

    elif args.command == "benchmark":
        from recall.eval.benchmark_index import (
            build_index_fingerprint,
            is_incomplete_index_slot,
            resolve_benchmark_index,
            save_benchmark_report,
            wipe_index,
        )
        from recall.eval.retrieval_benchmark import RetrievalBenchmarkRunner, _scale_to_limit, resolve_dataset
        from recall.storage import QdrantVectorStore

        dataset_path = Path(args.dataset_path) if args.dataset_path else None
        effective_limit = _scale_to_limit(args.scale) if args.scale else args.limit
        print(f"Loading dataset '{args.dataset}'...", flush=True)
        corpus = resolve_dataset(
            dataset=args.dataset,
            dataset_path=dataset_path,
            limit=args.limit,
            scale=args.scale,
            query_limit=args.query_limit,
            subsample_seed=args.subsample_seed,
        )
        print(
            f"Loaded {corpus.name}: {len(corpus.documents)} documents, "
            f"{len(corpus.queries)} queries, batch_size={args.batch_size}",
            flush=True,
        )

        async def _run_benchmark():
            from recall.embeddings import MockEmbeddingProvider, MockSparseEmbeddingProvider
            from recall.embeddings.fastembed_provider import FastEmbedProvider
            from recall.embeddings.fastembed_sparse_provider import FastEmbedSparseProvider

            config = load_config()
            index_context = None
            index_fingerprint = None
            collection_name = args.collection

            if not args.in_memory:
                index_root = (
                    Path(args.index_dir).expanduser()
                    if args.index_dir
                    else Path.home() / ".cache" / "recall" / "benchmark-indexes"
                )
                index_context = resolve_benchmark_index(
                    dataset=args.dataset,
                    document_limit=effective_limit,
                    scale=args.scale,
                    subsample_seed=args.subsample_seed,
                    fast=args.fast,
                    index_dir=index_root,
                )
                collection_name = index_context.collection_name
                if args.force_reindex:
                    wipe_index(index_context)
                    print(f"Cleared persisted index at {index_context.root_dir}", flush=True)
                elif is_incomplete_index_slot(index_context):
                    print(
                        f"Incomplete index slot (Qdrant data without manifest) — "
                        f"rebuilding at {index_context.root_dir}",
                        flush=True,
                    )
                    wipe_index(index_context)
                print(
                    f"Index slot (disk-backed, reused on next run): {index_context.root_dir}",
                    flush=True,
                )

            if args.fast:
                print(
                    "Fast scale mode: using lightweight mock embeddings (index/latency stress, not IR quality).",
                    flush=True,
                )
                dense_model = "mock-dense"
                sparse_model = "mock-sparse"
                if index_context is not None:
                    vector_store = QdrantVectorStore(path=str(index_context.qdrant_path))
                else:
                    vector_store = QdrantVectorStore(location=":memory:")
                service = RAGService(
                    vector_store=vector_store,
                    embedding_provider=MockEmbeddingProvider(dimensions=384),
                    sparse_embedder=MockSparseEmbeddingProvider(),
                    default_collection=collection_name,
                )
                if args.scale:
                    corpus.name = f"{corpus.name}+fast"
            else:
                if config.env.rag_env != "test":
                    print("Tip: set RAG_ENV=test for mock-embedding unit benchmarks.", flush=True)
                if index_context is not None:
                    vector_store = QdrantVectorStore(path=str(index_context.qdrant_path))
                    service = RAGService(vector_store=vector_store, default_collection=collection_name)
                else:
                    service = RAGService(default_collection=collection_name)
                dense_model = getattr(service.embedder, "model_name", "unknown-dense")
                sparse_model = getattr(service.sparse_embedder, "model_name", "unknown-sparse")
                if isinstance(service.embedder, FastEmbedProvider):
                    service.embedder.batch_size = args.batch_size
                if isinstance(service.sparse_embedder, FastEmbedSparseProvider):
                    service.sparse_embedder.batch_size = args.batch_size

            if index_context is not None:
                index_fingerprint = build_index_fingerprint(
                    dataset=args.dataset,
                    document_count=len(corpus.documents),
                    subsample_seed=args.subsample_seed,
                    scale=args.scale,
                    document_limit=effective_limit,
                    dense_model=dense_model,
                    sparse_model=sparse_model,
                    chunk_size=config.pipeline.ingest.chunk_size,
                    rag_mode=config.mode,
                    fast_mode=args.fast,
                )

            runner = RetrievalBenchmarkRunner(service=service)
            report = await runner.run(
                corpus=corpus,
                collection_name=collection_name,
                batch_size=args.batch_size,
                include_rerank=args.rerank,
                retrieve_limit=args.rerank_pool,
                index_context=index_context,
                index_fingerprint=index_fingerprint,
                reuse_index=not args.no_reuse_index,
            )
            markdown = report.to_markdown()
            print(markdown, flush=True)
            if args.output:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(markdown, encoding="utf-8")
                print(f"\nWrote report to {output_path}", flush=True)
            if index_context is not None:
                stamped, latest = save_benchmark_report(
                    index_context,
                    markdown,
                    rerank=args.rerank,
                )
                print(
                    f"\nPersisted benchmark report:\n"
                    f"  latest → {latest}\n"
                    f"  archive → {stamped}",
                    flush=True,
                )

        asyncio.run(_run_benchmark())


if __name__ == "__main__":
    main()
