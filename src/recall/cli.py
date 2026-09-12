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


if __name__ == "__main__":
    main()
