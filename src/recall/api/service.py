"""Unified RAG application service wiring ingestion, storage, retrieval, reranking, and synthesis."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from recall.adapters import ChunkingAdapterRegistry
from recall.core.config import AppConfig, load_config
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
from recall.synthesis.models import SynthesizedResponse

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
            location = self.config.env.qdrant_url or ":memory:"
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
                self.reranker = FlashRankReranker()

        # Initialize Compressor & Synthesizer
        self.compressor = compressor or ContextCompressor()
        if synthesizer:
            self.synthesizer = synthesizer
        else:
            active_syn = self.config.active_synthesis
            self.synthesizer = Synthesizer(
                model_name=active_syn.model or "ollama/llama3:8b",
                api_base=self.config.env.ollama_base_url if self.config.mode == "local" else None,
                api_key=self.config.env.openai_api_key,
                temperature=self.config.pipeline.synthesis.temperature,
                max_tokens=self.config.pipeline.synthesis.max_tokens,
            )

        self.adapter_registry = ChunkingAdapterRegistry()
        self._ensure_collection(self.default_collection)

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
        dense_embeddings = self.embedder.embed_texts(texts)
        sparse_embeddings = self.sparse_embedder.embed_texts(texts)
        for chunk, dense, sparse in zip(chunks, dense_embeddings, sparse_embeddings):
            chunk.embedding = dense
            chunk.sparse_vector = sparse

    def _index_sparse_legacy(self, collection_name: str, chunks: list[Chunk]) -> None:
        if not self.use_qdrant_sparse and self._legacy_sparse_indexes is not None:
            self._legacy_sparse_indexes.index(collection_name, chunks)

    async def ingest_file(
        self,
        file_path: Path,
        collection_name: str | None = None,
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

            self._attach_embeddings(chunks)

            upserted = self.vector_store.upsert(col, chunks)
            self._index_sparse_legacy(col, chunks)
            return upserted

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

            resolved_doc_id = doc_id or f"doc_{hash(source_uri) % 1000000:06d}"
            chunk = Chunk(
                id=f"text_{hash(text) % 10000000:07d}",
                text=text,
                metadata=ChunkMetadata(
                    doc_id=resolved_doc_id,
                    chunk_index=0,
                    source_uri=source_uri,
                ),
            )
            self._attach_embeddings([chunk])

            self.vector_store.upsert(col, [chunk])
            self._index_sparse_legacy(col, [chunk])
            return 1

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
        candidate_limit = retrieve_limit or max(limit * 4, self.config.pipeline.retrieval.top_k)
        candidates = await self.search(
            query=query,
            limit=candidate_limit,
            collection_name=col,
        )
        if not candidates:
            return []

        score_thresh = self.config.pipeline.retrieval.score_threshold
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

    async def query(
        self,
        question: str,
        collection_name: str | None = None,
        top_k: int = 5,
    ) -> SynthesizedResponse:
        """Executes the full end-to-end RAG pipeline:
        Retrieve -> Rerank -> Compress -> Synthesize with verified citations.
        """
        with trace_span(
            _tracer,
            "query.pipeline",
            {"collection.name": collection_name or self.default_collection, "top_k": top_k},
        ):
            candidates = await self.search(question, limit=top_k * 4, collection_name=collection_name)
            if not candidates:
                return SynthesizedResponse(
                    answer="Based on the provided documents, I do not have enough information to answer this question.",
                    citations=[],
                    unverified_citations=[],
                    model_name=getattr(self.synthesizer, "model_name", "unknown"),
                    latency_seconds=0.0,
                )

            score_thresh = getattr(self.config.pipeline.retrieval, "score_threshold", 0.35)
            with trace_span(_tracer, "rerank", {"candidate.count": len(candidates), "top_k": top_k}):
                reranked = self.reranker.rerank(
                    query=question,
                    candidates=candidates,
                    top_k=top_k,
                    score_threshold=score_thresh,
                )

            with trace_span(_tracer, "compress", {"candidate.count": len(reranked or candidates)}):
                compressed = self.compressor.compress(
                    query=question,
                    candidates=reranked if reranked else candidates[:top_k],
                    max_tokens_per_chunk=200,
                    max_total_tokens=1500,
                )

            response = await self.synthesizer.synthesize(
                query=question,
                candidates=compressed,
            )
            return response
