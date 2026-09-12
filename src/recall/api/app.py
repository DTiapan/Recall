"""Turnkey FastAPI web service exposing REST APIs and serving the embedded Web UI."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from recall.api.service import RAGService
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

    # Mount static assets
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    async def get_index() -> FileResponse:
        index_path = STATIC_DIR / "index.html"
        if not index_path.exists():
            raise HTTPException(status_code=404, detail="Web UI not found")
        return FileResponse(index_path)

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
        }

    @app.get("/v1/stats")
    async def get_stats(collection: str = "documents") -> dict[str, Any]:
        """Returns document count and vector storage metrics."""
        try:
            count = service.vector_store.count(collection)
        except Exception:
            count = service.sparse_index.count()

        return {
            "collection": collection,
            "total_chunks": count,
        }

    @app.post("/v1/ingest")
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
                indexed = await service.ingest_file(tmp_path, collection_name=collection)
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

    @app.post("/v1/chat", response_model=SynthesizedResponse)
    async def chat(request: ChatRequest) -> SynthesizedResponse:
        """Executes full RAG pipeline: Hybrid Search -> FlashRank -> Compression -> LiteLLM Synthesis."""
        response = await service.query(
            question=request.query,
            collection_name=request.collection,
            top_k=request.top_k,
        )
        return response

    return app
