# Lessons learned (LL)

Blameless capture of surprises, failed approaches, and reusable principles.

## Active index

| ID | Title | Status | Scope | Category |
|----|-------|--------|-------|----------|
| LL-001 | Benchmark labels must match indexed doc_id | resolved | universal | correctness |
| LL-002 | Batched ingest required for 10k+ eval scale | resolved | project | performance |
| LL-003 | FiQA@10k rerank regresses vs hybrid retrieval | resolved | project | correctness |
| LL-004 | Absolute cross-encoder thresholds break rerank eval | resolved | universal | correctness |
| LL-005 | Rerank wins are dataset-specific, not universal | open | universal | architecture |
| LL-006 | Custom metrics mislead vs BEIR community conventions | resolved | universal | process |

---

### LL-001 — Benchmark labels must match indexed doc_id

- **Date:** 2026-09-13
- **Status:** resolved
- **Scope:** universal
- **Promotion candidate:** yes
- **Category:** correctness
- **What happened:** 0% HitRate@5 on BEIR SciFact despite plausible retrieval
- **Hypothesis:** Embedding model mismatch
- **Evidence:** qrels referenced BEIR doc_ids; index used hashed source_uri
- **Root cause:** ingest_text replaced doc_id with convenience hash
- **Fix:** pass doc_id through ingest; relevant_sources fallback
- **General lesson:** Eval label keys must match indexed metadata — never substitute convenience hashes
- **Links:** DR-001, commit 5e0f296

### LL-002 — Batched ingest required for 10k+ eval scale

- **Date:** 2026-09-13
- **Status:** resolved
- **Scope:** project
- **Promotion candidate:** no
- **Category:** performance
- **What happened:** 10k FiQA run projected ~13 min at batch_size=1; user stopped early
- **Root cause:** Per-doc ONNX dispatch + Qdrant upsert round-trips
- **Fix:** ingest_chunks_batched batch_size=128; optional --fast for stress tiers
- **General lesson:** Scale benchmarks need batched embed+upsert in Recall's eval path
- **Links:** DR-002, commit 2ea19cb, AP-002

### LL-003 — FiQA@10k rerank regresses vs hybrid retrieval

- **Date:** 2026-09-13
- **Status:** resolved
- **Scope:** project
- **Promotion candidate:** no
- **Category:** correctness
- **What happened:** After 10k FiQA benchmark, rerank HitRate@5 **51.4%** vs hybrid **57.6%** (243-query subsample)
- **Root cause (confirmed):** `retrieval.score_threshold=0.35` applied only on rerank path; hybrid unfiltered
- **Fix:** `reranking.score_threshold: null` by default; MiniLM-L-12 wired from config; `candidate_k=50`
- **Evidence:** FiQA@10k fair subsample (648 queries) — hybrid **73.8%**, rerank **72.1%** (−1.7pp, not a bug)
- **General lesson:** Large rerank regressions on one path usually mean asymmetric filtering — measure both paths with identical gating before blaming the model
- **Links:** [fiqa-10k-ap003-rerank.md](../benchmarks/fiqa-10k-ap003-rerank.md), AP-003, LL-004

### LL-004 — Absolute cross-encoder thresholds break rerank eval

- **Date:** 2026-09-13
- **Status:** resolved
- **Scope:** universal
- **Promotion candidate:** yes
- **Category:** correctness
- **What happened:** Production RAG guides warn fixed cutoffs (e.g. 0.35) are uncalibrated; we apply `retrieval.score_threshold` only in `search_rerank()`, not hybrid `search()`
- **Root cause:** Threshold designed for synthesis gating, copied to reranker without per-query calibration
- **Fix:** Split `reranking.score_threshold` from `retrieval.score_threshold`; default null for rank-only eval
- **General lesson:** Use relative margin, z-score, or anchor normalization — never compare hybrid vs rerank when only one path is thresholded
- **Links:** AP-003, [investigation doc](../benchmarks/retrieval-quality-investigation.md)

### LL-005 — Rerank wins are dataset-specific, not universal

- **Date:** 2026-09-13
- **Status:** open
- **Scope:** universal
- **Promotion candidate:** yes
- **Category:** architecture
- **What happened:** Literature (rag-eval) shows rerank +0.026 nDCG on FiQA but negative delta on SciFact/NFCorpus
- **General lesson:** Enable rerank per domain after transfer eval; don't ship "rerank always on" as default
- **Links:** AP-003, ADR-008

### LL-006 — Custom metrics mislead vs BEIR community conventions

- **Date:** 2026-09-13
- **Status:** resolved
- **Scope:** universal
- **Promotion candidate:** yes
- **Category:** process
- **What happened:** We reported HitRate@5; BEIR community uses nDCG@10 / Recall@10 — hard to compare 57.6% to published 0.43 nDCG
- **Fix:** Added `pytrec_eval` (nDCG@10, Recall@10/100, MRR@10) to benchmark reports; fair qrels-aware subsample
- **General lesson:** Match benchmark community metrics (pytrec_eval) when citing external baselines; subsample methodology matters as much as metrics
- **Links:** AP-003 Phase B, FiQA@10k nDCG@10 **0.522**

---

## Archive
