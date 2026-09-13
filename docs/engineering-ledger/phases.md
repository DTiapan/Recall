# Phases

| Phase | Status | Gate evidence | Notes |
|-------|--------|---------------|-------|
| Shape | done | North-star in AGENTS.md | Enterprise RAG kit mission |
| Spec / ADR | done | ADR-013 + prior ADRs | Real-world benchmarks ADR |
| Build | done | 93 tests green (last full run) | Batched ingest landed 2ea19cb |
| Verify | in_progress | Sample 100% HitRate@5; SciFact 85.7% | FiQA 10k pending AP-002 |
| Ship | planned | CI green on main | GitHub Actions ci.yml |

## Phase log

- **2026-09-13** — Verify: sample + SciFact benchmarks documented in README; batched ingest committed
- **2026-09-13** — Build: DR-002 batched benchmark ingest batch_size=128
