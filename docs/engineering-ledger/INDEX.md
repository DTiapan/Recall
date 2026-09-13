# Engineering Ledger — Recall (RAG Kit)

> Read this file before non-trivial work. Update at end of each substantive session.

**Active phase:** Ship (in progress)  
**Last updated:** 2026-09-13

## Current focus

CI retrieval regression gate landed (`--min-hit-rate` / `--min-mrr` on `recall benchmark`). **Next:** AP-003 Phase C tuning (optional) or close Ship gate on green CI run.

## Open attack plan

- AP-003 Phase C — Reranker model comparison, hybrid weight grid, long-doc chunking ([research doc](../benchmarks/retrieval-quality-investigation.md))

## Recent sessions

### 2026-09-13 — AP-003 retrieval quality fixes + FiQA@10k re-run (complete)

**Phase:** Verify  
**Summary:** Fixed rerank path: split `reranking.score_threshold` (null default), wired `local_model` + `candidate_k=50`, fixed benchmark rerank metric aggregation, added `--rerank` CLI. Hardened disk persistence (`manifest.json`, `subsample.json`, auto-reports; incomplete slot detection). Re-ran FiQA@10k with fair subsample (648 queries): hybrid HR@5 **73.8%**, rerank **72.1%**, nDCG@10 **0.522**. Index at `~/.cache/recall/benchmark-indexes/beir-fiqa_10k_s42`. **109 tests green.**  
**Decisions:** DR-003, DR-004  
**Lessons:** LL-003 resolved, LL-006 resolved  
**Next:** Phase C tuning or Ship CI gate; reuse persisted index for fast re-eval  
**Docs:** [fiqa-10k-ap003-rerank.md](../benchmarks/fiqa-10k-ap003-rerank.md), [investigation](../benchmarks/retrieval-quality-investigation.md)

### 2026-09-13 — AP-002 FiQA@10k benchmark (complete)

**Phase:** Verify  
**Summary:** Ran `recall benchmark --dataset beir:fiqa --scale 10k --batch-size 128` with live 3-phase progress logging. 10k docs, 243 queries: HitRate@5 57.6%, ingest 6.7 docs/sec (~25 min), peak RSS 15.3 GB, query P99 324 ms. Report: docs/benchmarks/fiqa-10k.md; README table updated.  
**Next:** 100k `--fast` stress tier; streaming ingest for 1M+; Ship phase CI gate  
**Docs:** [benchmarks/README.md](../benchmarks/README.md) — results narrative + 10M roadmap

### 2026-09-13 — Craft adoption audit (complete)

**Phase:** Verify  
**Summary:** Cross-checked Recall against Craft v0.1 craft-adopt checklist. Ledger, ADRs, AGENTS.md, craft.project.yaml confirmed. Added docs/craft-setup.md for new-machine bootstrap; aligned CONSTRAINTS with using-craft + engineering-ledger.  
**Next:** Run AP-002 FiQA 10k; optional Battery onboard on new laptop

## Quick links

- [Craft setup (new machine)](../craft-setup.md)
- [Phases](phases.md)
- [Attack plans](attack-plans.md)
- [Decisions](decisions.md)
- [Lessons](lessons.md)
- [ADRs](../decisions/)
