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
    BaseSparseIndex,
    BaseSynthesizer,
    BaseVectorStore,
)
from recall.core.models import Chunk, ChunkMetadata, Document, IngestConfig, SearchResult
from recall.embeddings import FastEmbedProvider, MockEmbeddingProvider
from recall.preprocessing.ingest_gate import (
    DedupGateRegistry,
    document_from_file,
    should_ingest_document,
)
from recall.rerank import ContextCompressor, FlashRankReranker, MockReranker
from recall.retrieval import BM25Index, BM25IndexRegistry, HybridRetriever, RegistrySparseIndexAdapter
from recall.storage import QdrantVectorStore
from recall.synthesis import Synthesizer
from recall.synthesis.models import SynthesizedResponse

logger = logging.getLogger(__name__)


class RAGService:
    """Orchestrates all 5 RAG pipeline subsystems into a cohesive, turnkey service."""

    def __init__(
        self,
        config: AppConfig | None = None,
        vector_store: BaseVectorStore | None = None,
        embedding_provider: BaseEmbeddingProvider | None = None,
        sparse_index: BaseSparseIndex | BM25Index | BM25IndexRegistry | None = None,
        reranker: BaseReranker | None = None,
        compressor: BaseContextCompressor | None = None,
        synthesizer: BaseSynthesizer | None = None,
        dedup_gates: DedupGateRegistry | None = None,
        default_collection: str = "documents",
    ) -> None:
        self.config = config or load_config()
        self.default_collection = default_collection

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

        # Initialize per-collection sparse indexes
        if isinstance(sparse_index, BM25IndexRegistry):
            self.sparse_indexes = sparse_index
        elif isinstance(sparse_index, BM25Index):
            self.sparse_indexes = BM25IndexRegistry({default_collection: sparse_index})
        elif isinstance(sparse_index, RegistrySparseIndexAdapter):
            self.sparse_indexes = sparse_index._registry
        else:
            self.sparse_indexes = BM25IndexRegistry()

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

    def _get_retriever(self, collection_name: str) -> BaseHybridRetriever:
        timeout_sec = self.config.pipeline.retrieval.timeout_ms / 1000.0
        sparse_adapter = RegistrySparseIndexAdapter(self.sparse_indexes, collection_name)
        return HybridRetriever(
            vector_store=self.vector_store,
            embedding_provider=self.embedder,
            sparse_index=sparse_adapter,
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
            )

    def _should_ingest(self, document: Document, collection_name: str) -> bool:
        gate = self._dedup_gates.get(collection_name)
        return should_ingest_document(
            document,
            gate,
            enable_deduplication=self.config.pipeline.ingest.enable_deduplication,
        )

    async def ingest_file(
        self,
        file_path: Path,
        collection_name: str | None = None,
    ) -> int:
        """Parses, chunks, embeds, and indexes a file."""
        col = collection_name or self.default_collection
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

        # Generate dense embeddings
        texts = [c.searchable_text for c in chunks]
        embeddings = self.embedder.embed_texts(texts)
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        # Insert into Qdrant and append to collection-scoped BM25 index
        upserted = self.vector_store.upsert(col, chunks)
        self.sparse_indexes.index(col, chunks)
        return upserted

    async def ingest_text(
        self,
        text: str,
        source_uri: str = "manual_input.txt",
        collection_name: str | None = None,
    ) -> int:
        """Ingests raw text directly."""
        col = collection_name or self.default_collection
        self._ensure_collection(col)

        document = Document(content=text, source_uri=source_uri)
        if not self._should_ingest(document, col):
            logger.info("Skipping duplicate text ingest: %s", source_uri)
            return 0

        chunk = Chunk(
            id=f"text_{hash(text) % 10000000:07d}",
            text=text,
            metadata=ChunkMetadata(
                doc_id=f"doc_{hash(source_uri) % 1000000:06d}",
                chunk_index=0,
                source_uri=source_uri,
            ),
        )
        emb = self.embedder.embed_query(chunk.searchable_text)
        chunk.embedding = emb

        self.vector_store.upsert(col, [chunk])
        self.sparse_indexes.index(col, [chunk])
        return 1

    async def search(
        self,
        query: str,
        limit: int = 10,
        collection_name: str | None = None,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Performs concurrent hybrid retrieval (Dense + Sparse) with RRF."""
        col = collection_name or self.default_collection
        retriever = self._get_retriever(col)
        return await retriever.retrieve(query, limit=limit, filter_dict=filter_dict)

    async def query(
        self,
        question: str,
        collection_name: str | None = None,
        top_k: int = 5,
    ) -> SynthesizedResponse:
        """Executes the full end-to-end RAG pipeline:
        Retrieve -> Rerank -> Compress -> Synthesize with verified citations.
        """
        # 1. Retrieve candidates
        candidates = await self.search(question, limit=top_k * 4, collection_name=collection_name)
        if not candidates:
            return SynthesizedResponse(
                answer="Based on the provided documents, I do not have enough information to answer this question.",
                citations=[],
                unverified_citations=[],
                model_name=getattr(self.synthesizer, "model_name", "unknown"),
                latency_seconds=0.0,
            )

        # 2. Rerank candidates with cross-encoder
        score_thresh = getattr(self.config.pipeline.retrieval, "score_threshold", 0.35)
        reranked = self.reranker.rerank(
            query=question,
            candidates=candidates,
            top_k=top_k,
            score_threshold=score_thresh,
        )

        # 3. Compress context
        compressed = self.compressor.compress(
            query=question,
            candidates=reranked if reranked else candidates[:top_k],
            max_tokens_per_chunk=200,
            max_total_tokens=1500,
        )

        # 4. Synthesize answer with citation audit trail
        response = await self.synthesizer.synthesize(
            query=question,
            candidates=compressed,
        )
        return response
