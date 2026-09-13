"""Tests for qrels-aware BEIR subsampling."""

from recall.eval.datasets.subsample import count_evaluable_queries, select_beir_doc_ids


def test_select_beir_doc_ids_prefers_qrels_coverage():
    corpus_ids = [f"doc-{i}" for i in range(100)]
    qrels = {
        "q1": {"doc-0": 1, "doc-50": 1},
        "q2": {"doc-1": 1},
        "q3": {"doc-99": 1},
    }

    selected = select_beir_doc_ids(corpus_ids, qrels, limit=10, seed=7)
    assert len(selected) == 10
    assert "doc-0" in selected
    assert "doc-1" in selected
    assert "doc-99" in selected
    assert count_evaluable_queries(qrels, set(selected)) == 3


def test_select_beir_doc_ids_is_seeded():
    corpus_ids = [f"doc-{i}" for i in range(20)]
    qrels = {"q1": {"doc-0": 1}}

    a = select_beir_doc_ids(corpus_ids, qrels, limit=5, seed=42)
    b = select_beir_doc_ids(corpus_ids, qrels, limit=5, seed=42)
    c = select_beir_doc_ids(corpus_ids, qrels, limit=5, seed=99)

    assert a == b
    assert a != c
