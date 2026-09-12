# Project Guidelines: Production Enterprise RAG Kit

## Mission & Scope
We are building a **production-grade, enterprise-scale Retrieval-Augmented Generation (RAG) framework & toolkit**.
Designed for open-source adoption as a modular, plug-and-play library/service capable of scaling up to **10 Million+ documents** while handling critical edge cases with rigorous engineering standards.

## Architecture Principles
1. **Spec & ADR First**: Every architectural fork (chunking strategy, vector DB selection, indexing mode, hybrid retrieval weighting, reranking pipeline) MUST be documented via an Architecture Decision Record (ADR) in `docs/decisions/` before lock-in.
2. **Phase-by-Phase Modular Design**:
   - **Phase 1: Ingestion & Document Preprocessing Engine**: Multi-format document loading, semantic / contextual awareness chunking vs. recursive token chunking, metadata enrichment, deduplication, and benchmark evaluations.
   - **Phase 2: Storage & Vector Indexing Layer**: Production vector database abstraction (e.g. Qdrant / Milvus), index tuning (HNSW vs. IVF-PQ/SQ) tailored for 10M+ scale, filtering, and memory/disk storage footprint optimization.
   - **Phase 3: Hybrid Retrieval & Fusion**: Two-fold retrieval combining Lexical (BM25 / Sparse) + Dense Semantic (Bi-Encoder embeddings) with Reciprocal Rank Fusion (RRF) / weighted score fusion.
   - **Phase 4: Reranking & Quality Filter**: Cross-encoder rerankers, threshold gating, contextual compression, and noise elimination.
   - **Phase 5: Synthesis & Generation Guardrails**: Context assembly, prompt injection prevention, hallucination checks, cite attribution, and observability.
3. **Pluggable & Extensible**: Clear interfaces/contracts so external teams can swap chunkers, embedding models, vector stores, and rerankers without modifying internal engine logic.
4. **Scale & Edge Cases**:
   - Designed for 10M+ documents with memory efficiency, batching, async ingestion queues, and distributed indexing considerations.
   - Robust error recovery, rate limiting, token window truncation, and multi-modal readiness.

## Engineering Disciplines & Skill Workflows
This repository leverages `addyosmani/agent-skills`:
- **ADRs**: `skills/documentation-and-adrs` for all architectural trade-offs in `docs/decisions/` (`ADR-001-...`).
- **Brainstorming & Ideation**: `skills/idea-refine` and `skills/interview-me` for vetting architectural trade-offs.
- **Specs & Task Planning**: `skills/spec-driven-development` and `skills/planning-and-task-breakdown`.
- **TDD & Code Health**: `skills/test-driven-development`, `skills/code-review-and-quality`, `skills/code-simplification`.
- **Performance & Observability**: `skills/performance-optimization`, `skills/observability-and-instrumentation`.
