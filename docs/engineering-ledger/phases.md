# Phases

| Phase | Status | Gate evidence | Notes |
|-------|--------|---------------|-------|
| Shape | done | North-star in AGENTS.md | Enterprise RAG kit mission |
| Spec / ADR | done | ADR-013 + prior ADRs | Real-world benchmarks ADR |
| Build | done | 109 tests green | Batched ingest, rerank config wiring |
| Verify | in_progress | Sample 100%; SciFact@500 fair 91.0% HR@5; FiQA@10k fair 73.8% HR@5, nDCG@10 0.522 | Disk-persisted indexes; AP-003 A+B done |
| Ship | planned | CI green on main | GitHub Actions ci.yml |

## Phase log

- **2026-09-13** — Verify: AP-003 Phase A+B — rerank fixes, fair subsample, disk index persistence; FiQA@10k 73.8% hybrid / 72.1% rerank on 648 queries
- **2026-09-13** — Verify: FiQA@10k benchmark complete (AP-002); 6.7 docs/sec, 15.3 GB peak RSS, report at docs/benchmarks/fiqa-10k.md
- **2026-09-13** — Verify: sample + SciFact benchmarks documented in README; batched ingest committed
- **2026-09-13** — Build: DR-002 batched benchmark ingest batch_size=128
