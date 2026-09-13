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
        help="Maximum labeled queries to evaluate (BEIR datasets, default: 50)",
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
        from recall.eval.retrieval_benchmark import RetrievalBenchmarkRunner, resolve_dataset

        dataset_path = Path(args.dataset_path) if args.dataset_path else None
        corpus = resolve_dataset(
            dataset=args.dataset,
            dataset_path=dataset_path,
            limit=args.limit,
            scale=args.scale,
            query_limit=args.query_limit,
        )

        async def _run_benchmark():
            from recall.embeddings import MockEmbeddingProvider, MockSparseEmbeddingProvider
            from recall.embeddings.fastembed_provider import FastEmbedProvider
            from recall.embeddings.fastembed_sparse_provider import FastEmbedSparseProvider

            config = load_config()
            if args.fast:
                print(
                    "Fast scale mode: using lightweight mock embeddings (index/latency stress, not IR quality).",
                    flush=True,
                )
                service = RAGService(
                    embedding_provider=MockEmbeddingProvider(dimensions=384),
                    sparse_embedder=MockSparseEmbeddingProvider(),
                    default_collection=args.collection,
                )
                if args.scale:
                    corpus.name = f"{corpus.name}+fast"
            else:
                if config.env.rag_env != "test":
                    print("Tip: set RAG_ENV=test for mock-embedding unit benchmarks.", flush=True)
                service = RAGService(default_collection=args.collection)
                if isinstance(service.embedder, FastEmbedProvider):
                    service.embedder.batch_size = args.batch_size
                if isinstance(service.sparse_embedder, FastEmbedSparseProvider):
                    service.sparse_embedder.batch_size = args.batch_size

            runner = RetrievalBenchmarkRunner(service=service)
            report = await runner.run(
                corpus=corpus,
                collection_name=args.collection,
                batch_size=args.batch_size,
            )
            print(report.to_markdown(), flush=True)
            if args.output:
                output_path = Path(args.output)
                output_path.write_text(report.to_markdown(), encoding="utf-8")
                print(f"\nWrote report to {output_path}")

        asyncio.run(_run_benchmark())


if __name__ == "__main__":
    main()
