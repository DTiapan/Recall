"""Utilities for converting sparse embeddings to Qdrant-compatible structures."""

from __future__ import annotations

from typing import Any

from qdrant_client.http import models


def sparse_dict_to_qdrant(sparse_vector: dict[Any, float] | None) -> models.SparseVector | None:
    """Converts a token-indexed sparse weight map into a Qdrant SparseVector."""
    if not sparse_vector:
        return None

    indices: list[int] = []
    values: list[float] = []
    for key, weight in sparse_vector.items():
        if not weight:
            continue
        indices.append(int(key))
        values.append(float(weight))

    if not indices:
        return None

    return models.SparseVector(indices=indices, values=values)
