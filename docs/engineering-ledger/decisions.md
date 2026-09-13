# Tactical decisions (DR)

Decisions reversible without a formal ADR. Promote to `docs/decisions/` when reversal cost is high.

## Active index

| ID | Title | Status | Date |
|----|-------|--------|------|
| DR-001 | Preserve BEIR doc_id through ingest | accepted | 2026-09-13 |
| DR-002 | Batched benchmark ingest at 128 | accepted | 2026-09-13 |
| DR-003 | Split rerank score_threshold from retrieval | accepted | 2026-09-13 |
| DR-004 | Disk-persisted benchmark index slots | accepted | 2026-09-13 |

---

### DR-001 — Preserve BEIR doc_id through ingest

- **Date:** 2026-09-13
- **Status:** accepted
- **Context:** SciFact benchmark returned 0% HitRate@5 — qrels could not match indexed chunks
- **Options:**
  1. Hash source_uri as doc_id (status quo)
  2. Pass BEIR doc_id through ingest_text + relevant_sources fallback
- **Decision:** Option 2
- **Rationale:** Benchmark labels must match indexed metadata; hashing broke qrel join
- **Tradeoffs accepted:** Ingest API must accept explicit doc_id for eval paths
- **Links:** commit 5e0f296, LL-001

### DR-002 — Batched benchmark ingest at 128

- **Date:** 2026-09-13
- **Status:** accepted
- **Context:** 10k FiQA ingest ~13 min at batch_size=1 — ONNX dispatch + Qdrant round-trip bound
- **Options:**
  1. batch embed+upsert (batch_size=128)
  2. `--fast` mock mode for 100k+ stress only
  3. Both
- **Decision:** Both — default batch_size=128; `--fast` for scale stress
- **Rationale:** Batching removes per-doc overhead; fast mode for throughput experiments only
- **Tradeoffs accepted:** Benchmark ingest bypasses dedup at scale
- **Links:** commit 2ea19cb, ADR-013, AP-002

### DR-003 — Split rerank score_threshold from retrieval

- **Date:** 2026-09-13
- **Status:** accepted
- **Context:** FiQA@10k rerank HR@5 51.4% vs hybrid 57.6%; `retrieval.score_threshold` filtered rerank only
- **Decision:** Add `reranking.score_threshold` (default null); wire `reranking.local_model` to FlashRank via `resolve_flashrank_model_name()`
- **Rationale:** Fair eval + config honesty; absolute cutoff remains opt-in after calibration
- **Links:** LL-003, LL-004, AP-003

### DR-004 — Disk-persisted benchmark index slots

- **Date:** 2026-09-13
- **Status:** accepted
- **Context:** FiQA@10k re-runs re-embedded from scratch (~25 min); partial Qdrant dirs without manifest could not reuse
- **Decision:** Default `recall benchmark` writes to `~/.cache/recall/benchmark-indexes/<slug>/` with `manifest.json` fingerprint, `subsample.json` doc IDs, and `reports/latest_*.md`; auto-wipe incomplete slots (Qdrant without manifest)
- **Rationale:** Scale evals are expensive; index is the condensed artifact worth keeping; manifest enables skip-reingest on matching fingerprint
- **Tradeoffs accepted:** Disk footprint (~GB per 10k slot); `--in-memory` / `--force-reindex` for CI or clean rebuilds
- **Links:** AP-003, [benchmarks README](../benchmarks/README.md)

---

## Archive
