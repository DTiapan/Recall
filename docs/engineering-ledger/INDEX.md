# Engineering Ledger — Recall (RAG Kit)

> Read this file before non-trivial work. Update at end of each substantive session.

**Active phase:** Verify  
**Last updated:** 2026-09-13

## Current focus

Recall Phase 1 (ingestion + eval) is largely built. Current emphasis: **real-world benchmark evidence** (sample corpus, BEIR SciFact, batched scale ingest) and Craft adoption for running narrative alongside ADRs.

## Open attack plan

- AP-002 — 10k FiQA scale benchmark with batched ingest (planned)

## Recent sessions

### 2026-09-13 — Craft adoption + ledger backfill

**Phase:** Verify  
**Summary:** Adopted Craft orchestration layer on Recall. Backfilled ledger for benchmark/eval work (sample corpus, BEIR doc_id fix, batched ingest). Tagged universal lesson on benchmark metadata alignment.  
**Decisions:** DR-001, DR-002  
**Lessons:** LL-001 (universal), LL-002 (project)  
**Next:** Run AP-002 FiQA 10k with batch_size=128; document in README

## Quick links

- [Phases](phases.md)
- [Attack plans](attack-plans.md)
- [Decisions](decisions.md)
- [Lessons](lessons.md)
- [ADRs](../decisions/)
