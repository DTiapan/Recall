"""Retrieval and fusion engine for Recall."""

from recall.retrieval.bm25 import BM25Index, tokenize_lexical
from recall.retrieval.fusion import reciprocal_rank_fusion
from recall.retrieval.hybrid import HybridRetriever
from recall.retrieval.sparse_registry import BM25IndexRegistry, RegistrySparseIndexAdapter

__all__ = [
    "BM25Index",
    "BM25IndexRegistry",
    "RegistrySparseIndexAdapter",
    "tokenize_lexical",
    "reciprocal_rank_fusion",
    "HybridRetriever",
]
