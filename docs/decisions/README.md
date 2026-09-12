# Architecture Decision Records (ADRs)

This directory maintains the Architecture Decision Records for the Production RAG project, following the [documentation-and-adrs](file:///.agents/skills/documentation-and-adrs/SKILL.md) skill convention.

## Index of Decisions

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [ADR-001](0001-record-architecture-decisions.md) | Record Architecture Decisions | Accepted | 2026-09-13 |
| [ADR-002](0002-production-rag-architecture.md) | Production RAG Architecture, Query Transformation & Security | Accepted | 2026-09-13 |
| [ADR-003](0003-ingestion-and-chunking-strategy.md) | Ingestion Preprocessing, Deduplication & Chunking Strategy | Accepted | 2026-09-13 |
| [ADR-004](0004-format-aware-chunking-adapters.md) | Format-Aware Chunking Adapters & Structural Ingestion Engine | Accepted | 2026-09-13 |
| [ADR-005](0005-turnkey-enterprise-rag-platform.md) | Turnkey Enterprise Architecture, Dual-Mode Engine & OSS Stack | Accepted | 2026-09-13 |
| [ADR-006](0006-vector-storage-and-indexing.md) | Unified Qdrant Vector Store, Dense/Sparse Storage & FastEmbed | Accepted | 2026-09-13 |
| *Upcoming* | Hybrid Retrieval, Circuit Breakers & Reciprocal Rank Fusion (Phase 3) | Proposed | Pending |

## ADR Lifecycle
```
PROPOSED → ACCEPTED → (SUPERSEDED or DEPRECATED)
```
Do not delete old ADRs. When a decision evolves, create a new record that references and supersedes the predecessor.
