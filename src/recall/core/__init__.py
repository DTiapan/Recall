"""Core domain models, interfaces, and configuration engine."""

from recall.core.config import AppConfig, EnvSettings, PipelineConfig, load_config
from recall.core.interfaces import (
    BaseChunker,
    BaseContextEnricher,
    BaseDeduplicator,
    BaseDocumentLoader,
    BaseEmbeddingProvider,
    BaseHybridRetriever,
    BaseSparseIndex,
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
    "BaseContextEnricher",
    "BaseDeduplicator",
    "BaseDocumentLoader",
    "BaseEmbeddingProvider",
    "BaseHybridRetriever",
    "BaseSparseIndex",
    "BaseVectorStore",
    "Chunk",
    "ChunkMetadata",
    "Document",
    "IngestConfig",
    "SearchResult",
    "compute_content_hash",
]
