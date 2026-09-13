# Architecture Decision Records (ADRs)

This directory maintains the Architecture Decision Records for the Production RAG project, following the [documentation-and-adrs](file:///.agents/skills/documentation-and-adrs/SKILL.md) skill convention.

**Scale language in ADRs:** References to 10M+ documents describe **design targets and rationale**, not validated production scale. Published benchmark evidence lives in [benchmarks/README.md](../benchmarks/README.md) (largest proven run: FiQA@10k, Sep 2026).

## Index of Decisions

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [ADR-001](0001-record-architecture-decisions.md) | Record Architecture Decisions | Accepted | 2026-09-13 |
| [ADR-002](0002-production-rag-architecture.md) | Production RAG Architecture, Query Transformation & Security | Accepted | 2026-09-13 |
| [ADR-003](0003-ingestion-and-chunking-strategy.md) | Ingestion Preprocessing, Deduplication & Chunking Strategy | Accepted | 2026-09-13 |
| [ADR-004](0004-format-aware-chunking-adapters.md) | Format-Aware Chunking Adapters & Structural Ingestion Engine | Accepted | 2026-09-13 |
| [ADR-005](0005-turnkey-enterprise-rag-platform.md) | Turnkey Enterprise Architecture, Dual-Mode Engine & OSS Stack | Accepted | 2026-09-13 |
| [ADR-006](0006-vector-storage-and-indexing.md) | Unified Qdrant Vector Store, Dense/Sparse Storage & FastEmbed | Accepted | 2026-09-13 |
| [ADR-007](0007-hybrid-retrieval-and-fusion.md) | Concurrent Hybrid Retrieval, Reciprocal Rank Fusion & Circuit Breakers | Accepted | 2026-09-13 |
| [ADR-008](0008-reranking-and-quality-filtering.md) | Cross-Encoder Reranking, Threshold Gating & Context Compression | Accepted | 2026-09-13 |
| [ADR-009](0009-synthesis-and-guardrails.md) | Context Sandboxing, Injection Defense, LiteLLM & Citation Verification | Accepted | 2026-09-13 |
| [ADR-010](0010-turnkey-rest-api-and-web-ui.md) | Turnkey REST API, Embedded Web UI & CLI Tooling | Accepted | 2026-09-13 |
| [ADR-011](0011-packaging-and-production-hardening.md) | Multi-Stage Hardened Dockerfile & Compose Orchestration | Accepted | 2026-09-13 |
| [ADR-012](0012-synthetic-corpus-and-scale-benchmarking.md) | Synthetic Enterprise Corpus Generation & Multi-Scale Stress Benchmarking | Accepted (partial — generator not built) | 2026-09-13 |
| [ADR-013](0013-real-world-benchmark-datasets.md) | Real-World Benchmark Datasets (BEIR + Curated Corpus) | Accepted | 2026-09-13 |

## ADR Lifecycle
```
PROPOSED → ACCEPTED → (SUPERSEDED or DEPRECATED)
```
Do not delete old ADRs. When a decision evolves, create a new record that references and supersedes the predecessor.
