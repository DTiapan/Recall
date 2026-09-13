"""BEIR-standard metrics via pytrec_eval (same engine BEIR's EvaluateRetrieval uses)."""

from __future__ import annotations


def compute_beir_metrics(
    qrels: dict[str, dict[str, int]],
    results: dict[str, dict[str, float]],
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """Return mean nDCG@k, Recall@k, and MRR (recip_rank) for the evaluated query set."""
    if not qrels or not results:
        return {}

    try:
        import pytrec_eval
    except ImportError as exc:
        raise ImportError(
            "BEIR metrics require pytrec_eval: uv pip install -e '.[benchmark]'"
        ) from exc

    ks = k_values or [10]
    measure_names = {f"ndcg_cut_{k}" for k in ks} | {f"recall_{k}" for k in ks} | {"recip_rank"}

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, measure_names)
    scores = evaluator.evaluate(results)
    if not scores:
        return {}

    totals = {f"ndcg@{k}": 0.0 for k in ks}
    totals.update({f"recall@{k}": 0.0 for k in ks})
    totals["mrr@10"] = 0.0

    for query_scores in scores.values():
        for k in ks:
            totals[f"ndcg@{k}"] += float(query_scores.get(f"ndcg_cut_{k}", 0.0))
            totals[f"recall@{k}"] += float(query_scores.get(f"recall_{k}", 0.0))
        totals["mrr@10"] += float(query_scores.get("recip_rank", 0.0))

    count = len(scores)
    return {key: round(value / count, 5) for key, value in totals.items()}


def filter_qrels_to_docs(
    qrels: dict[str, dict[str, int]],
    allowed_doc_ids: set[str],
) -> dict[str, dict[str, int]]:
    """Keep only positive judgments for documents present in the subsample."""
    filtered: dict[str, dict[str, int]] = {}
    for query_id, rels in qrels.items():
        kept = {
            doc_id: score
            for doc_id, score in rels.items()
            if doc_id in allowed_doc_ids and score > 0
        }
        if kept:
            filtered[query_id] = kept
    return filtered
