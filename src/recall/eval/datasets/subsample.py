"""Reproducible BEIR corpus subsampling with qrels-aware document selection."""

from __future__ import annotations

import random
from collections.abc import Iterable


def select_beir_doc_ids(
    corpus_doc_ids: Iterable[str],
    qrels: dict[str, dict[str, int]],
    limit: int | None,
    seed: int = 42,
) -> list[str]:
    """Select up to ``limit`` document IDs for a BEIR subsample.

    Strategy (borrowed from BEIR reproducibility practice — seed + qrels coverage):

    1. Greedily pick judged documents that maximize **query coverage** until the cap
       is reached or every qrels-referenced doc is included.
    2. Fill any remaining slots with a **seeded shuffle** of other corpus docs.

    This avoids the "first N dict keys" bias that left FiQA@500 with only 17/648 queries.
    """
    ordered_ids = list(corpus_doc_ids)
    if limit is None or limit >= len(ordered_ids):
        return ordered_ids

    corpus_set = set(ordered_ids)
    rng = random.Random(seed)

    # doc_id -> query ids it satisfies
    doc_to_queries: dict[str, set[str]] = {}
    for query_id, rels in qrels.items():
        for doc_id, score in rels.items():
            if score <= 0 or doc_id not in corpus_set:
                continue
            doc_to_queries.setdefault(doc_id, set()).add(query_id)

    selected: list[str] = []
    covered_queries: set[str] = set()

    judged_pool = list(doc_to_queries.keys())
    while judged_pool and len(selected) < limit:
        best_doc = max(
            judged_pool,
            key=lambda doc_id: len(doc_to_queries[doc_id] - covered_queries),
        )
        gain = doc_to_queries[best_doc] - covered_queries
        if not gain and len(selected) + len(judged_pool) > limit:
            # No marginal coverage; break to random fill for reproducibility.
            break
        selected.append(best_doc)
        judged_pool.remove(best_doc)
        covered_queries |= doc_to_queries[best_doc]

    remaining = [doc_id for doc_id in ordered_ids if doc_id not in set(selected)]
    rng.shuffle(remaining)
    selected.extend(remaining[: limit - len(selected)])
    return selected


def count_evaluable_queries(
    qrels: dict[str, dict[str, int]],
    allowed_doc_ids: set[str],
) -> int:
    """Count queries with at least one relevant doc in ``allowed_doc_ids``."""
    total = 0
    for rels in qrels.values():
        if any(doc_id in allowed_doc_ids and score > 0 for doc_id, score in rels.items()):
            total += 1
    return total
