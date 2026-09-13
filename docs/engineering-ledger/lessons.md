# Lessons learned (LL)

Blameless capture of surprises, failed approaches, and reusable principles.

## Active index

| ID | Title | Status | Scope | Category |
|----|-------|--------|-------|----------|
| LL-001 | Benchmark labels must match indexed doc_id | resolved | universal | correctness |
| LL-002 | Batched ingest required for 10k+ eval scale | resolved | project | performance |

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

---

## Archive
