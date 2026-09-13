"""Turnkey FastAPI web service exposing REST APIs and serving the embedded Web UI."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from recall.api.auth import build_api_key_guard
from recall.api.service import RAGService
from recall.observability import get_tracer, init_tracing
from recall.core.models import SearchResult
from recall.synthesis.models import SynthesizedResponse

STATIC_DIR = Path(__file__).parent / "static"


class SearchRequest(BaseModel):
    query: str = Field(..., description="Query search string")
    limit: int = Field(default=10, ge=1, le=100)
    collection: str = Field(default="documents")


class ChatRequest(BaseModel):
    query: str = Field(..., description="User question")
    collection: str = Field(default="documents")
    top_k: int = Field(default=5, ge=1, le=20)
    stream: bool = Field(default=False, description="Stream answer tokens via SSE")
    model: str | None = Field(default=None, description="Allowlisted OpenRouter model id")
    rerank: bool | None = Field(default=None, description="Override reranking; false skips FlashRank")


def _configure_observability(service: RAGService) -> bool:
    config = service.config
    tracing_configured = (
        config.pipeline.observability.enabled and not config.env.otel_sdk_disabled
    )
    service_name = config.env.otel_service_name or config.pipeline.observability.service_name
    should_initialize = tracing_configured and config.env.rag_env != "test"
    init_tracing(
        service_name=service_name,
        enabled=should_initialize,
        otlp_endpoint=config.env.otel_exporter_otlp_endpoint,
    )
    return tracing_configured


def create_app(rag_service: RAGService | None = None) -> FastAPI:
    """Factory creating and configuring the turnkey Recall FastAPI application."""
    app = FastAPI(
        title="Recall Enterprise RAG",
        description="The Open-Source, Turnkey Enterprise Retrieval-Augmented Generation Platform",
        version="0.1.0",
    )

    # Enable CORS for microservice interoperability
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    service = rag_service or RAGService()
    tracing_enabled = _configure_observability(service)
    verify_api_key = build_api_key_guard(service.config.env.rag_api_key)

    @app.middleware("http")
    async def trace_requests(request: Request, call_next):
        tracer = get_tracer("recall.api")
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        with tracer.start_as_current_span("http.request") as span:
            span.set_attribute("http.method", request.method)
            span.set_attribute("http.route", request.url.path)
            span.set_attribute("request.id", request_id)
            response = await call_next(request)
            span.set_attribute("http.status_code", response.status_code)
            response.headers["x-request-id"] = request_id
            return response

    # Mount static assets
    if STATIC_DIR.exists():
        static_app = StaticFiles(directory=str(STATIC_DIR))

        @app.get("/static/{file_path:path}", include_in_schema=False)
        async def static_assets(file_path: str):
            target = (STATIC_DIR / file_path).resolve()
            if not str(target).startswith(str(STATIC_DIR.resolve())) or not target.is_file():
                raise HTTPException(status_code=404)
            return FileResponse(
                target,
                headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
            )

        app.mount("/static", static_app, name="static")

    @app.get("/", include_in_schema=False)
    async def get_index() -> FileResponse:
        index_path = STATIC_DIR / "index.html"
        if not index_path.exists():
            raise HTTPException(status_code=404, detail="Web UI not found")
        return FileResponse(
            index_path,
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )

    @app.get("/v1/health")
    async def health_check() -> dict[str, Any]:
        """Returns health status, active collection, and embedding dimensionality."""
        return {
            "status": "healthy",
            "service": "recall-kit",
            "version": "0.1.0",
            "mode": str(service.config.mode),
            "embedding_dimensions": service.embedder.dimensions,
            "default_collection": service.default_collection,
            "tracing_enabled": tracing_enabled,
            "model": getattr(service.synthesizer, "model_name", "unknown"),
            "model_display": service.model_display_name(),
        }

    @app.get("/v1/config")
    async def get_ui_config() -> dict[str, Any]:
        """Returns UI-facing runtime configuration (models, mode, collection, pipeline defaults)."""
        return service.get_ui_config()

    @app.delete("/v1/documents", dependencies=[Depends(verify_api_key)])
    async def delete_document(
        source_uri: str,
        collection: str = "documents",
    ) -> dict[str, Any]:
        """Deletes all chunks indexed under the given source_uri."""
        if not source_uri:
            raise HTTPException(status_code=400, detail="source_uri is required")
        deleted = service.delete_document(source_uri=source_uri, collection_name=collection)
        return {
            "status": "deleted",
            "source_uri": source_uri,
            "chunks_deleted": deleted,
            "collection": collection,
        }

    @app.get("/v1/documents")
    async def list_documents(collection: str = "documents") -> dict[str, Any]:
        """Lists indexed documents grouped by source_uri with chunk counts."""
        docs = service.list_documents(collection_name=collection)
        total_chunks = sum(doc["chunk_count"] for doc in docs)
        return {
            "collection": collection,
            "documents": docs,
            "document_count": len(docs),
            "total_chunks": total_chunks,
        }

    @app.get("/v1/stats")
    async def get_stats(collection: str = "documents") -> dict[str, Any]:
        """Returns document count and vector storage metrics."""
        try:
            dense_count = service.vector_store.count(collection)
        except Exception:
            dense_count = 0

        sparse_count = service.sparse_chunk_count(collection)

        return {
            "collection": collection,
            "total_chunks": max(dense_count, sparse_count),
            "dense_chunks": dense_count,
            "sparse_chunks": sparse_count,
        }

    @app.post("/v1/ingest", dependencies=[Depends(verify_api_key)])
    async def ingest_document(
        file: UploadFile | None = File(None),
        text: str | None = Form(None),
        source_uri: str | None = Form(None),
        collection: str = Form("documents"),
    ) -> dict[str, Any]:
        """Ingests a document file or raw text into vector storage and lexical sparse index."""
        if file is not None:
            filename = file.filename or "upload.tmp"
            suffix = Path(filename).suffix

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                shutil.copyfileobj(file.file, tmp)
                tmp_path = Path(tmp.name)

            try:
                indexed = await service.ingest_file(
                    tmp_path,
                    collection_name=collection,
                    source_uri=filename,
                )
                return {
                    "status": "indexed",
                    "filename": filename,
                    "chunks_indexed": indexed,
                    "collection": collection,
                }
            finally:
                if tmp_path.exists():
                    os.unlink(tmp_path)

        elif text:
            indexed = await service.ingest_text(
                text=text,
                source_uri=source_uri or "manual_input.txt",
                collection_name=collection,
            )
            return {
                "status": "indexed",
                "chunks_indexed": indexed,
                "collection": collection,
            }

        else:
            raise HTTPException(status_code=400, detail="Must provide either a file upload or text payload")

    @app.post("/v1/search", response_model=list[SearchResult])
    async def search(request: SearchRequest) -> list[SearchResult]:
        """Performs concurrent hybrid search (Dense HNSW + BM25+) with Reciprocal Rank Fusion."""
        results = await service.search(
            query=request.query,
            limit=request.limit,
            collection_name=request.collection,
        )
        return results

    @app.post("/v1/chat", dependencies=[Depends(verify_api_key)])
    async def chat(request: ChatRequest):
        """Executes full RAG pipeline: Hybrid Search -> FlashRank -> Compression -> LiteLLM Synthesis."""
        try:
            service.validate_model_id(request.model)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if request.stream:
            async def event_stream():
                async for event in service.query_stream(
                    question=request.query,
                    collection_name=request.collection,
                    top_k=request.top_k,
                    model=request.model,
                    rerank=request.rerank,
                ):
                    yield f"data: {json.dumps(event)}\n\n"
                    await asyncio.sleep(0)

            return StreamingResponse(
                event_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        response = await service.query(
            question=request.query,
            collection_name=request.collection,
            top_k=request.top_k,
            model=request.model,
            rerank=request.rerank,
        )
        return response

    return app
