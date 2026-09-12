"""Per-collection sparse index registry for isolated lexical search."""

from __future__ import annotations

from recall.core.interfaces import BaseSparseIndex
from recall.core.models import Chunk, SearchResult
from recall.retrieval.bm25 import BM25Index
from typing import Any


class BM25IndexRegistry:
    """Maps collection names to isolated BM25 indexes."""

    def __init__(self, indexes: dict[str, BM25Index] | None = None) -> None:
        self._indexes: dict[str, BM25Index] = indexes or {}

    def get(self, collection_name: str) -> BM25Index:
        if collection_name not in self._indexes:
            self._indexes[collection_name] = BM25Index()
        return self._indexes[collection_name]

    def index(self, collection_name: str, chunks: list[Chunk]) -> int:
        return self.get(collection_name).index(chunks)

    def search(
        self,
        collection_name: str,
        query: str,
        limit: int = 10,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        return self.get(collection_name).search(query, limit=limit, filter_dict=filter_dict)

    def count(self, collection_name: str) -> int:
        return self.get(collection_name).count()

    def collections(self) -> list[str]:
        return list(self._indexes.keys())


class RegistrySparseIndexAdapter(BaseSparseIndex):
    """Adapts a BM25IndexRegistry to the BaseSparseIndex protocol for a fixed collection."""

    def __init__(self, registry: BM25IndexRegistry, collection_name: str) -> None:
        self.registry = registry
        self.collection_name = collection_name

    @property
    def _registry(self) -> BM25IndexRegistry:
        return self.registry

    @property
    def _collection_name(self) -> str:
        return self.collection_name

    def index(self, chunks: list[Chunk]) -> int:
        return self._registry.index(self._collection_name, chunks)

    def search(
        self,
        query: str,
        limit: int = 10,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        return self._registry.search(
            self._collection_name,
            query,
            limit=limit,
            filter_dict=filter_dict,
        )

    def count(self) -> int:
        return self._registry.count(self._collection_name)
