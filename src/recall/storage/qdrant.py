"""Qdrant Vector Store implementation supporting in-memory mode, scalar quantization, and payload filtering."""

from __future__ import annotations

import uuid
from typing import Any
from qdrant_client import QdrantClient
from qdrant_client.http import models

from recall.core.interfaces import BaseVectorStore
from recall.core.models import Chunk, ChunkMetadata, SearchResult


DISTANCE_MAP: dict[str, models.Distance] = {
    "Cosine": models.Distance.COSINE,
    "Dot": models.Distance.DOT,
    "Euclid": models.Distance.EUCLID,
}


def _chunk_id_to_uuid(chunk_id: str) -> str:
    """Generates a deterministic RFC 4122 UUID from an arbitrary chunk string identifier."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


class QdrantVectorStore:
    """Production vector database adapter for Qdrant.
    Supports in-memory testing (location=':memory:') and production Docker/cloud endpoints.
    """

    def __init__(
        self,
        client: QdrantClient | None = None,
        location: str | None = None,
        url: str | None = None,
        api_key: str | None = None,
        path: str | None = None,
    ) -> None:
        if client is not None:
            self.client = client
        elif url is not None:
            self.client = QdrantClient(url=url, api_key=api_key)
        elif path is not None:
            self.client = QdrantClient(path=path)
        else:
            # Default to in-memory mode for zero-setup execution
            self.client = QdrantClient(location=location or ":memory:")

    def collection_exists(self, collection_name: str) -> bool:
        return self.client.collection_exists(collection_name=collection_name)

    def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: str = "Cosine",
        enable_quantization: bool = True,
        hnsw_m: int = 16,
        hnsw_ef_construct: int = 100,
    ) -> None:
        q_distance = DISTANCE_MAP.get(distance, models.Distance.COSINE)

        quant_config: models.QuantizationConfig | None = None
        if enable_quantization:
            quant_config = models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True,
                )
            )

        vectors_config = {
            "dense": models.VectorParams(
                size=vector_size,
                distance=q_distance,
                hnsw_config=models.HnswConfigDiff(m=hnsw_m, ef_construct=hnsw_ef_construct),
                quantization_config=quant_config,
            )
        }

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=vectors_config,
        )

        # Create payload index for fast filtering on server Qdrant
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for field_name in ["doc_id", "file_type", "policy_name", "content_type"]:
                try:
                    self.client.create_payload_index(
                        collection_name=collection_name,
                        field_name=field_name,
                        field_schema=models.PayloadSchemaType.KEYWORD,
                    )
                except Exception:
                    pass


    def upsert(self, collection_name: str, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0

        points: list[models.PointStruct] = []
        for c in chunks:
            if not c.embedding:
                continue

            point_id = _chunk_id_to_uuid(c.id)
            payload = {
                "chunk_id": c.id,
                "text": c.text,
                "contextualized_text": c.contextualized_text,
                "doc_id": c.metadata.doc_id,
                "chunk_index": c.metadata.chunk_index,
                "total_chunks": c.metadata.total_chunks,
                "token_count": c.metadata.token_count,
                "page_number": c.metadata.page_number,
                "total_pages": c.metadata.total_pages,
                "file_type": c.metadata.file_type,
                "content_type": c.metadata.content_type,
                "section_hierarchy": c.metadata.section_hierarchy,
                "policy_name": c.metadata.policy_name,
                "source_uri": c.metadata.source_uri,
                "extra": c.metadata.extra,
            }

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector={"dense": c.embedding},
                    payload=payload,
                )
            )

        if points:
            self.client.upsert(collection_name=collection_name, points=points)
        return len(points)

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 10,
        filter_dict: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        query_filter: models.Filter | None = None
        if filter_dict:
            conditions = []
            for k, v in filter_dict.items():
                conditions.append(
                    models.FieldCondition(
                        key=k,
                        match=models.MatchValue(value=v),
                    )
                )
            query_filter = models.Filter(must=conditions)

        # Qdrant client 1.19+ supports query_points or search
        response = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            using="dense",
            limit=limit,
            query_filter=query_filter,
            score_threshold=score_threshold,
            with_payload=True,
        )

        results: list[SearchResult] = []
        for pt in response.points:
            p = pt.payload or {}
            meta = ChunkMetadata(
                doc_id=p.get("doc_id", ""),
                chunk_index=p.get("chunk_index", 0),
                total_chunks=p.get("total_chunks", 1),
                token_count=p.get("token_count", 0),
                page_number=p.get("page_number"),
                total_pages=p.get("total_pages"),
                file_type=p.get("file_type"),
                content_type=p.get("content_type", "text"),
                section_hierarchy=p.get("section_hierarchy", []),
                policy_name=p.get("policy_name"),
                source_uri=p.get("source_uri"),
                extra=p.get("extra", {}),
            )

            results.append(
                SearchResult(
                    chunk_id=p.get("chunk_id", str(pt.id)),
                    score=float(pt.score) if pt.score is not None else 0.0,
                    text=p.get("text", ""),
                    metadata=meta,
                    vector_name="dense",
                )
            )

        return results

    def count(self, collection_name: str) -> int:
        return self.client.count(collection_name=collection_name, exact=True).count

    def delete_collection(self, collection_name: str) -> None:
        self.client.delete_collection(collection_name=collection_name)
