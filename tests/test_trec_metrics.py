"""Tests for pytrec_eval BEIR metric wrapper."""

from recall.eval.trec_metrics import compute_beir_metrics


def test_compute_beir_metrics_perfect_run():
    qrels = {
        "q1": {"d1": 1, "d2": 0},
        "q2": {"d3": 1},
    }
    results = {
        "q1": {"d1": 1.0, "d9": 0.1},
        "q2": {"d3": 0.9, "d8": 0.2},
    }

    metrics = compute_beir_metrics(qrels, results, k_values=[10])

    assert metrics["ndcg@10"] == 1.0
    assert metrics["recall@10"] == 1.0
    assert metrics["mrr@10"] == 1.0
