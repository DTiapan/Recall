# Attack plans

## Active index

| ID | Title | Status |
|----|-------|--------|
| AP-002 | FiQA 10k scale benchmark (batched) | done |
| AP-003 | Retrieval quality investigation & fixes | in_progress (A+B done, C open) |

---

### AP-001 — Real-world eval + sample corpus (backfill)

- **Date:** 2026-09-01 – 2026-09-13
- **Status:** done
- **Goal:** Prove retrieval quality on real formats and BEIR SciFact
- **Problem framing:** Synthetic evals insufficient; need BEIR + PDF/DOCX sample corpus with rerank metrics
- **Components to attack:**
  1. Sample corpus (MD/PDF/DOCX) + queries
  2. BEIR SciFact loader with doc_id preservation
  3. Rerank metrics + config.yaml local defaults
  4. CI gate + README benchmark table
  5. Batched ingest for scale (batch_size=128, --fast mode)
- **Done when:** Tests green; README table; SciFact HitRate@5 > 80%
- **Links:** commits 096a041, 5e0f296, 96ed9f1, 2ea19cb; ADR-013

### AP-002 — FiQA 10k scale benchmark (batched)

- **Date:** 2026-09-13
- **Status:** done
- **Goal:** Document throughput/RSS/P99 at 10k docs with batched ingest
- **Components to attack:**
  1. `recall benchmark --dataset beir:fiqa --scale 10k --batch-size 128`
  2. Update README benchmark table
  3. Record evidence in phases.md
- **Done when:** Benchmark completes; README + ledger updated
- **Links:** DR-002

### AP-003 — Retrieval quality investigation & fixes

- **Date:** 2026-09-13
- **Status:** in_progress
- **Goal:** Fix confirmed bugs, align eval with BEIR conventions, establish fair quality baselines before scale stress
- **Research:** [retrieval-quality-investigation.md](../benchmarks/retrieval-quality-investigation.md)
- **Components to attack:**
  1. **Phase A (correctness):** ✅ Wire reranker config; `candidate_k=50`; split `score_threshold`; `--rerank` CLI; disk index persistence; FiQA@10k re-run
  2. **Phase B (measurement):** ✅ pytrec_eval; qrels-aware subsample; SciFact@500 + FiQA@500 + FiQA@10k fair reports
  3. **Phase C (tuning):** Reranker model comparison; hybrid weight grid; long-doc chunking for BEIR; per-dataset rerank toggle
- **Done when:** Rerank ≥ hybrid on FiQA OR documented opt-out; BEIR-comparable metrics in README; investigation doc updated with before/after
- **FiQA@10k post-fix (2026-09-13):** hybrid HR@5 **73.8%**, rerank **72.1%** (−1.7pp, no threshold regression); nDCG@10 **0.522**; index persisted at `~/.cache/recall/benchmark-indexes/beir-fiqa_10k_s42`
- **Links:** LL-003, LL-004, LL-005, LL-006

---

## Archive

<!-- Move cancelled/superseded AP summaries here -->
