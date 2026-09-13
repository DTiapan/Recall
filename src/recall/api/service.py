"""Unified RAG application service wiring ingestion, storage, retrieval, reranking, and synthesis."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from collections.abc import AsyncIterator, Callable
from typing import Any

from recall.adapters import ChunkingAdapterRegistry
from recall.core.config import AppConfig, load_config
from recall.core.model_display import model_display_name
from recall.core.interfaces import (
    BaseContextCompressor,
    BaseEmbeddingProvider,
    BaseHybridRetriever,
    BaseReranker,
    BaseSparseEmbeddingProvider,
    BaseSparseIndex,
    BaseSynthesizer,
    BaseVectorStore,
)
from recall.core.models import Chunk, ChunkMetadata, Document, IngestConfig, SearchResult
from recall.embeddings import FastEmbedProvider, MockEmbeddingProvider
from recall.embeddings.fastembed_sparse_provider import FastEmbedSparseProvider
from recall.embeddings.mock_sparse_provider import MockSparseEmbeddingProvider
from recall.preprocessing.ingest_gate import (
    DedupGateRegistry,
    document_from_file,
    should_ingest_document,
)
from recall.rerank import ContextCompressor, FlashRankReranker, MockReranker
from recall.retrieval import HybridRetriever
from recall.retrieval.bm25 import BM25Index
from recall.retrieval.sparse_registry import BM25IndexRegistry, RegistrySparseIndexAdapter
from recall.storage import QdrantVectorStore
from recall.storage.qdrant_sparse_index import QdrantSparseIndex
from recall.observability import get_tracer, trace_span
from recall.synthesis import Synthesizer
from recall.synthesis.citations import extract_and_verify_citations
from recall.synthesis.models import PipelineTiming, SynthesizedResponse
from recall.synthesis.sandbox import build_sandboxed_context

logger = logging.getLogger(__name__)
_tracer = get_tracer("recall.service")


class RAGService:
    """Orchestrates all 5 RAG pipeline subsystems into a cohesive, turnkey service."""

    def __init__(
        self,
        config: AppConfig | None = None,
        vector_store: BaseVectorStore | None = None,
        embedding_provider: BaseEmbeddingProvider | None = None,
        sparse_embedder: BaseSparseEmbeddingProvider | None = None,
        sparse_index: BaseSparseIndex | BM25Index | BM25IndexRegistry | None = None,
        reranker: BaseReranker | None = None,
        compressor: BaseContextCompressor | None = None,
        synthesizer: BaseSynthesizer | None = None,
        dedup_gates: DedupGateRegistry | None = None,
        default_collection: str = "documents",
        use_qdrant_sparse: bool = True,
    ) -> None:
        self.config = config or load_config()
        self.default_collection = default_collection
        self.use_qdrant_sparse = use_qdrant_sparse

        # Initialize Embedding Provider
        if embedding_provider:
            self.embedder = embedding_provider
        else:
            if self.config.env.rag_env == "test":
                self.embedder = MockEmbeddingProvider(dimensions=16)
            else:
                model_name = self.config.active_embedding.model_name or "BAAI/bge-small-en-v1.5"
                self.embedder = FastEmbedProvider(model_name=model_name)

        # Initialize Vector Store
        if vector_store:
            self.vector_store = vector_store
        else:
            qdrant_path = self.config.env.qdrant_path
            qdrant_url = self.config.env.qdrant_url
            if qdrant_path:
                self.vector_store = QdrantVectorStore(path=qdrant_path)
            elif qdrant_url and qdrant_url.startswith(("http://", "https://")):
                self.vector_store = QdrantVectorStore(
                    url=qdrant_url,
                    api_key=self.config.env.qdrant_api_key,
                )
            else:
                location = qdrant_url or ":memory:"
                self.vector_store = QdrantVectorStore(
                    location=location,
                    api_key=self.config.env.qdrant_api_key,
                )

        # Initialize sparse embedding + index strategy
        if sparse_embedder:
            self.sparse_embedder = sparse_embedder
        elif self.config.env.rag_env == "test":
            self.sparse_embedder = MockSparseEmbeddingProvider()
        else:
            self.sparse_embedder = FastEmbedSparseProvider()

        self._legacy_sparse_indexes: BM25IndexRegistry | None = None
        if isinstance(sparse_index, BM25IndexRegistry):
            self._legacy_sparse_indexes = sparse_index
            self.use_qdrant_sparse = False
        elif isinstance(sparse_index, BM25Index):
            self._legacy_sparse_indexes = BM25IndexRegistry({default_collection: sparse_index})
            self.use_qdrant_sparse = False
        elif isinstance(sparse_index, RegistrySparseIndexAdapter):
            self._legacy_sparse_indexes = sparse_index.registry
            self.use_qdrant_sparse = False

        self._dedup_gates = dedup_gates or DedupGateRegistry(
            near_dup_threshold=self.config.pipeline.ingest.near_dup_threshold,
        )

        # Initialize Reranker
        if reranker:
            self.reranker = reranker
        else:
            if self.config.env.rag_env == "test":
                self.reranker = MockReranker()
            else:
                from recall.rerank.flashrank import resolve_flashrank_model_name

                rerank_model = resolve_flashrank_model_name(
                    self.config.pipeline.reranking.local_model
                )
                self.reranker = FlashRankReranker(model_name=rerank_model)

        # Initialize Compressor & Synthesizer
        self.compressor = compressor or ContextCompressor()
        if synthesizer:
            self.synthesizer = synthesizer
        else:
            model_name, api_base, api_key = self._resolve_synthesis_llm()
            self.synthesizer = Synthesizer(
                model_name=model_name,
                api_base=api_base,
                api_key=api_key,
                temperature=self.config.pipeline.synthesis.temperature,
                max_tokens=self.config.pipeline.synthesis.max_tokens,
            )

        self.adapter_registry = ChunkingAdapterRegistry()
        self._ensure_collection(self.default_collection)

    def _resolve_synthesis_llm(
        self,
        model_override: str | None = None,
    ) -> tuple[str, str | None, str | None]:
        """Pick LLM endpoint: OpenRouter (if key set) > cloud profile > local Ollama/LM Studio."""
        env = self.config.env
        if env.openrouter_api_key:
            model = (
                model_override
                or env.local_llm_model
                or self.config.pipeline.ui.default_model
                or "openai/gpt-oss-120b"
            )
            if not model.startswith("openrouter/"):
                model = f"openrouter/{model}"
            return model, None, env.openrouter_api_key

        active_syn = self.config.active_synthesis
        if self.config.mode == "local":
            model = (
                model_override
                or env.local_llm_model
                or active_syn.model
                or "ollama/llama3:8b"
            )
            return model, env.ollama_base_url, env.openai_api_key or "lm-studio"

        model = model_override or active_syn.model or "gpt-4o"
        return model, None, env.openai_api_key

    def _get_sparse_index(self, collection_name: str) -> BaseSparseIndex:
        if self.use_qdrant_sparse:
            if not isinstance(self.vector_store, QdrantVectorStore):
                raise TypeError("Qdrant sparse retrieval requires a QdrantVectorStore instance.")
            return QdrantSparseIndex(
                vector_store=self.vector_store,
                collection_name=collection_name,
                sparse_embedder=self.sparse_embedder,
            )

        if self._legacy_sparse_indexes is None:
            self._legacy_sparse_indexes = BM25IndexRegistry()
        return RegistrySparseIndexAdapter(self._legacy_sparse_indexes, collection_name)

    def _rerank_candidate_limit(self, output_limit: int) -> int:
        """Hybrid retrieval depth before cross-encoder reranking."""
        return max(self.config.pipeline.reranking.candidate_k, output_limit)

    def _get_retriever(self, collection_name: str) -> BaseHybridRetriever:
        timeout_sec = self.config.pipeline.retrieval.timeout_ms / 1000.0
        return HybridRetriever(
            vector_store=self.vector_store,
            embedding_provider=self.embedder,
            sparse_index=self._get_sparse_index(collection_name),
            collection_name=collection_name,
            dense_weight=self.config.pipeline.retrieval.dense_weight,
            sparse_weight=self.config.pipeline.retrieval.sparse_weight,
            rrf_k=self.config.pipeline.retrieval.rrf_k,
            dense_timeout_seconds=timeout_sec,
        )

    def _ensure_collection(self, collection_name: str) -> None:
        """Ensures Qdrant collection exists with the appropriate vector dimension."""
        if not self.vector_store.collection_exists(collection_name):
            enable_quant = getattr(self.config.pipeline.storage.quantization, "enabled", True)
            self.vector_store.create_collection(
                collection_name=collection_name,
                vector_size=self.embedder.dimensions,
                enable_quantization=enable_quant,
                enable_sparse=self.use_qdrant_sparse,
            )

    def _should_ingest(self, document: Document, collection_name: str) -> bool:
        gate = self._dedup_gates.get(collection_name)
        return should_ingest_document(
            document,
            gate,
            enable_deduplication=self.config.pipeline.ingest.enable_deduplication,
        )

    def _attach_embeddings(self, chunks: list[Chunk]) -> None:
        texts = [c.searchable_text for c in chunks]
        with ThreadPoolExecutor(max_workers=2) as pool:
            dense_future = pool.submit(self.embedder.embed_texts, texts)
            sparse_future = pool.submit(self.sparse_embedder.embed_texts, texts)
            dense_embeddings = dense_future.result()
            sparse_embeddings = sparse_future.result()
        for chunk, dense, sparse in zip(chunks, dense_embeddings, sparse_embeddings):
            chunk.embedding = dense
            chunk.sparse_vector = sparse

    def _index_sparse_legacy(self, collection_name: str, chunks: list[Chunk]) -> None:
        if not self.use_qdrant_sparse and self._legacy_sparse_indexes is not None:
            self._legacy_sparse_indexes.index(collection_name, chunks)

    def build_text_chunk(
        self,
        text: str,
        source_uri: str,
        doc_id: str | None = None,
        chunk_index: int = 0,
    ) -> Chunk:
        """Builds a single-chunk document payload without embedding or indexing."""
        resolved_doc_id = doc_id or f"doc_{hash(source_uri) % 1000000:06d}"
        return Chunk(
            id=f"text_{resolved_doc_id}_{chunk_index}",
            text=text,
            metadata=ChunkMetadata(
                doc_id=resolved_doc_id,
                chunk_index=chunk_index,
                source_uri=source_uri,
            ),
        )

    def prepare_file_chunks(self, file_path: Path) -> list[Chunk]:
        """Parses and chunks a file without embedding or indexing."""
        ingest_config = IngestConfig(
            chunk_size=self.config.pipeline.ingest.chunk_size,
            chunk_overlap=self.config.pipeline.ingest.chunk_overlap,
            enable_deduplication=self.config.pipeline.ingest.enable_deduplication,
        )
        return self.adapter_registry.process(file_path, config=ingest_config)

    async def ingest_chunks_batched(
        self,
        chunks: list[Chunk],
        collection_name: str | None = None,
        batch_size: int = 128,
        on_batch: Callable[[int, int], None] | None = None,
    ) -> int:
        """Embed and upsert chunks in vectorized micro-batches for high-throughput ingest."""
        col = collection_name or self.default_collection
        with trace_span(
            _tracer,
            "ingest.batch",
            {"collection.name": col, "chunk.count": len(chunks), "batch.size": batch_size},
        ):
            self._ensure_collection(col)
            if not chunks:
                return 0

            total = 0
            for start in range(0, len(chunks), batch_size):
                batch = chunks[start : start + batch_size]
                self._attach_embeddings(batch)
                total += self.vector_store.upsert(col, batch)
                self._index_sparse_legacy(col, batch)
                if on_batch is not None:
                    on_batch(min(start + len(batch), len(chunks)), len(chunks))
            return total

    async def ingest_file(
        self,
        file_path: Path,
        collection_name: str | None = None,
        source_uri: str | None = None,
    ) -> int:
        """Parses, chunks, embeds, and indexes a file."""
        col = collection_name or self.default_collection
        with trace_span(
            _tracer,
            "ingest.file",
            {"collection.name": col, "file.path": str(file_path)},
        ):
            self._ensure_collection(col)

            if not self._should_ingest(document_from_file(file_path), col):
                logger.info("Skipping duplicate document: %s", file_path)
                return 0

            ingest_config = IngestConfig(
                chunk_size=self.config.pipeline.ingest.chunk_size,
                chunk_overlap=self.config.pipeline.ingest.chunk_overlap,
                enable_deduplication=self.config.pipeline.ingest.enable_deduplication,
            )

            chunks = self.adapter_registry.process(file_path, config=ingest_config)
            if not chunks:
                return 0

            if source_uri:
                for chunk in chunks:
                    chunk.metadata.source_uri = source_uri

            return await self.ingest_chunks_batched(chunks, collection_name=col)

    async def ingest_text(
        self,
        text: str,
        source_uri: str = "manual_input.txt",
        collection_name: str | None = None,
        doc_id: str | None = None,
    ) -> int:
        """Ingests raw text directly."""
        col = collection_name or self.default_collection
        with trace_span(
            _tracer,
            "ingest.text",
            {"collection.name": col, "source.uri": source_uri},
        ):
            self._ensure_collection(col)

            document = Document(content=text, source_uri=source_uri)
            if not self._should_ingest(document, col):
                logger.info("Skipping duplicate text ingest: %s", source_uri)
                return 0

            chunk = self.build_text_chunk(text=text, source_uri=source_uri, doc_id=doc_id)
            await self.ingest_chunks_batched([chunk], collection_name=col)
            return 1

    def list_documents(self, collection_name: str | None = None) -> list[dict[str, Any]]:
        """Returns indexed documents grouped by source_uri with chunk counts."""
        from recall.core.source_uri import is_junk_source_uri

        col = collection_name or self.default_collection
        if isinstance(self.vector_store, QdrantVectorStore):
            docs = self.vector_store.list_sources(col)
            return [doc for doc in docs if not is_junk_source_uri(doc.get("source_uri"))]
        return []

    def default_model_slug(self) -> str:
        """OpenRouter-style slug for the server default synthesis model."""
        if self.config.pipeline.ui.default_model:
            return self.config.pipeline.ui.default_model
        if self.config.env.local_llm_model:
            return self.config.env.local_llm_model.removeprefix("openrouter/")
        raw = getattr(self.synthesizer, "model_name", "unknown")
        return raw.removeprefix("openrouter/")

    def allowed_ui_models(self) -> list[dict[str, str]]:
        """Models exposed to the Web UI dropdown."""
        configured = self.config.pipeline.ui.models
        if configured:
            return [{"id": m.id, "label": m.label} for m in configured]
        slug = self.default_model_slug()
        return [{"id": slug, "label": model_display_name(slug)}]

    def validate_model_id(self, model_id: str | None) -> str | None:
        """Raises ValueError when model_id is not on the UI allowlist."""
        if model_id is None:
            return None
        allowed = {m["id"] for m in self.allowed_ui_models()}
        if model_id not in allowed:
            raise ValueError(f"Model '{model_id}' is not in the configured allowlist")
        return model_id

    def resolve_synthesizer(self, model_id: str | None = None) -> Synthesizer:
        """Build a per-request synthesizer for the given allowlisted model id."""
        validated = self.validate_model_id(model_id)
        slug = validated or self.default_model_slug()
        model_name, api_base, api_key = self._resolve_synthesis_llm(slug)
        return Synthesizer(
            model_name=model_name,
            api_base=api_base,
            api_key=api_key,
            temperature=self.config.pipeline.synthesis.temperature,
            max_tokens=self.config.pipeline.synthesis.max_tokens,
        )

    def _synthesizer_for_request(self, model_id: str | None) -> Synthesizer:
        """Use the startup synthesizer by default; build per-request when model is overridden."""
        if model_id is None:
            return self.synthesizer
        return self.resolve_synthesizer(model_id)

    def model_display_name(self, model_id: str | None = None) -> str:
        """Human-readable label for a synthesis model."""
        if model_id:
            for entry in self.allowed_ui_models():
                if entry["id"] == model_id:
                    return entry["label"]
            return model_display_name(model_id)
        raw = getattr(self.synthesizer, "model_name", "unknown")
        if raw.startswith("ollama/"):
            return raw.removeprefix("ollama/")
        return model_display_name(raw)

    def get_ui_config(self) -> dict[str, Any]:
        """Runtime configuration payload for the embedded Web UI."""
        slug = self.default_model_slug()
        return {
            "mode": str(self.config.mode),
            "model": getattr(self.synthesizer, "model_name", "unknown"),
            "model_display": self.model_display_name(),
            "active_model": slug,
            "default_model": slug,
            "models": self.allowed_ui_models(),
            "collection": self.default_collection,
            "embedding_dimensions": self.embedder.dimensions,
            "auth_required": bool(self.config.env.rag_api_key),
            "top_k_default": self.config.pipeline.reranking.top_n,
            "rerank_enabled": self.config.pipeline.reranking.enabled,
            "candidate_k": self.config.pipeline.reranking.candidate_k,
        }

    def sparse_chunk_count(self, collection_name: str) -> int:
        if self.use_qdrant_sparse:
            try:
                if not self.vector_store.collection_exists(collection_name):
                    return 0
                return self.vector_store.count(collection_name)
            except Exception:
                return 0
        if self._legacy_sparse_indexes is None:
            return 0
        return self._legacy_sparse_indexes.count(collection_name)

    async def search(
        self,
        query: str,
        limit: int = 10,
        collection_name: str | None = None,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Performs concurrent hybrid retrieval (Dense + Sparse) with RRF."""
        col = collection_name or self.default_collection
        with trace_span(
            _tracer,
            "retrieve.hybrid",
            {"collection.name": col, "query.length": len(query), "retrieve.limit": limit},
        ):
            retriever = self._get_retriever(col)
            return await retriever.retrieve(query, limit=limit, filter_dict=filter_dict)

    async def search_rerank(
        self,
        query: str,
        limit: int = 5,
        collection_name: str | None = None,
        retrieve_limit: int | None = None,
    ) -> list[SearchResult]:
        """Hybrid retrieval followed by cross-encoder reranking."""
        col = collection_name or self.default_collection
        candidate_limit = retrieve_limit or self._rerank_candidate_limit(limit)
        candidates = await self.search(
            query=query,
            limit=candidate_limit,
            collection_name=col,
        )
        if not candidates:
            return []

        score_thresh = self.config.pipeline.reranking.score_threshold
        with trace_span(
            _tracer,
            "rerank",
            {"candidate.count": len(candidates), "top_k": limit},
        ):
            reranked = self.reranker.rerank(
                query=query,
                candidates=candidates,
                top_k=limit,
                score_threshold=score_thresh,
            )
        return reranked if reranked else candidates[:limit]

    async def _prepare_query(
        self,
        question: str,
        collection_name: str | None,
        top_k: int,
        rerank: bool | None = None,
    ) -> tuple[list[SearchResult] | None, PipelineTiming, SynthesizedResponse | None]:
        """Retrieve, rerank, and compress candidates. Returns early response when no hits."""
        rerank_enabled = rerank if rerank is not None else self.config.pipeline.reranking.enabled
        retrieve_limit = self._rerank_candidate_limit(top_k) if rerank_enabled else top_k

        retrieve_start = time.perf_counter()
        candidates = await self.search(
            question,
            limit=retrieve_limit,
            collection_name=collection_name,
        )
        retrieve_ms = (time.perf_counter() - retrieve_start) * 1000.0

        if not candidates:
            timing = PipelineTiming(retrieve_ms=round(retrieve_ms, 2), total_ms=round(retrieve_ms, 2))
            return [], timing, None

        final_candidates = candidates[:top_k]
        rerank_ms = 0.0
        if rerank_enabled:
            score_thresh = self.config.pipeline.reranking.score_threshold
            rerank_start = time.perf_counter()
            with trace_span(
                _tracer,
                "rerank",
                {"candidate.count": len(candidates), "top_k": top_k},
            ):
                reranked = self.reranker.rerank(
                    query=question,
                    candidates=candidates,
                    top_k=top_k,
                    score_threshold=score_thresh,
                )
            rerank_ms = (time.perf_counter() - rerank_start) * 1000.0
            final_candidates = reranked if reranked else candidates[:top_k]

        compress_start = time.perf_counter()
        with trace_span(_tracer, "compress", {"candidate.count": len(final_candidates)}):
            compressed = self.compressor.compress(
                query=question,
                candidates=final_candidates,
                max_tokens_per_chunk=200,
                max_total_tokens=1500,
            )
        compress_ms = (time.perf_counter() - compress_start) * 1000.0

        timing = PipelineTiming(
            retrieve_ms=round(retrieve_ms, 2),
            rerank_ms=round(rerank_ms, 2),
            compress_ms=round(compress_ms, 2),
        )
        return compressed, timing, None

    def delete_document(self, source_uri: str, collection_name: str | None = None) -> int:
        """Remove all chunks indexed under source_uri. Returns deleted point count."""
        col = collection_name or self.default_collection
        if isinstance(self.vector_store, QdrantVectorStore):
            return self.vector_store.delete_by_source_uri(col, source_uri)
        return 0

    async def query(
        self,
        question: str,
        collection_name: str | None = None,
        top_k: int = 5,
        model: str | None = None,
        rerank: bool | None = None,
    ) -> SynthesizedResponse:
        """Executes the full end-to-end RAG pipeline:
        Retrieve -> Rerank -> Compress -> Synthesize with verified citations.
        """
        pipeline_start = time.perf_counter()
        with trace_span(
            _tracer,
            "query.pipeline",
            {"collection.name": collection_name or self.default_collection, "top_k": top_k},
        ):
            synthesizer = self._synthesizer_for_request(model)
            compressed, timing, _early = await self._prepare_query(
                question, collection_name, top_k, rerank=rerank,
            )

            synthesis_start = time.perf_counter()
            response = await synthesizer.synthesize(
                query=question,
                candidates=compressed,
            )
            synthesis_ms = (time.perf_counter() - synthesis_start) * 1000.0
            total_ms = (time.perf_counter() - pipeline_start) * 1000.0

            return response.model_copy(
                update={
                    "model_name": synthesizer.model_name,
                    "latency_seconds": round(total_ms / 1000.0, 4),
                    "timing": timing.model_copy(
                        update={
                            "synthesis_ms": round(synthesis_ms, 2),
                            "total_ms": round(total_ms, 2),
                        }
                    ),
                }
            )

    async def query_stream(
        self,
        question: str,
        collection_name: str | None = None,
        top_k: int = 5,
        model: str | None = None,
        rerank: bool | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Streams RAG pipeline events: status, token deltas, and a final done payload."""
        pipeline_start = time.perf_counter()
        with trace_span(
            _tracer,
            "query.pipeline",
            {"collection.name": collection_name or self.default_collection, "top_k": top_k},
        ):
            synthesizer = self._synthesizer_for_request(model)
            yield {"event": "status", "phase": "retrieving"}
            compressed, timing, _early = await self._prepare_query(
                question, collection_name, top_k, rerank=rerank,
            )

            yield {"event": "status", "phase": "generating"}
            synthesis_start = time.perf_counter()
            answer_parts: list[str] = []
            async for delta in synthesizer.synthesize_stream(
                query=question,
                candidates=compressed,
            ):
                answer_parts.append(delta)
                yield {"event": "token", "delta": delta}

            answer = "".join(answer_parts)
            synthesis_ms = (time.perf_counter() - synthesis_start) * 1000.0
            total_ms = (time.perf_counter() - pipeline_start) * 1000.0
            _, candidate_map = build_sandboxed_context(compressed)
            verified, unverified = extract_and_verify_citations(answer, candidate_map)
            final_timing = timing.model_copy(
                update={
                    "synthesis_ms": round(synthesis_ms, 2),
                    "total_ms": round(total_ms, 2),
                }
            )

            yield {
                "event": "done",
                "answer": answer,
                "citations": [c.model_dump() for c in verified],
                "unverified_citations": unverified,
                "model_name": synthesizer.model_name,
                "latency_seconds": round(total_ms / 1000.0, 4),
                "timing": final_timing.model_dump(),
            }
