"""Core domain models, interfaces, and configuration engine."""

from recall.core.config import AppConfig, EnvSettings, PipelineConfig, load_config
from recall.core.interfaces import (
    BaseChunker,
    BaseContextCompressor,
    BaseContextEnricher,
    BaseDeduplicator,
    BaseDocumentLoader,
    BaseEmbeddingProvider,
    BaseHybridRetriever,
    BaseReranker,
    BaseSparseIndex,
    BaseSynthesizer,
    BaseVectorStore,
)
from recall.core.models import (
    Chunk,
    ChunkMetadata,
    Document,
    IngestConfig,
    SearchResult,
    compute_content_hash,
)

__all__ = [
    "AppConfig",
    "EnvSettings",
    "PipelineConfig",
    "load_config",
    "BaseChunker",
    "BaseContextCompressor",
    "BaseContextEnricher",
    "BaseDeduplicator",
    "BaseDocumentLoader",
    "BaseEmbeddingProvider",
    "BaseHybridRetriever",
    "BaseReranker",
    "BaseSparseIndex",
    "BaseSynthesizer",
    "BaseVectorStore",
    "Chunk",
    "ChunkMetadata",
    "Document",
    "IngestConfig",
    "SearchResult",
    "compute_content_hash",
]
