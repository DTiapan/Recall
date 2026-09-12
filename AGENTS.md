# Project Guidelines: Production Enterprise RAG Kit

## Mission & North Star
We are building a **turnkey, production-grade, enterprise-scale Retrieval-Augmented Generation (RAG) platform & toolkit**.
Designed for open-source adoption by small-to-medium businesses (SMBs) and enterprise engineering teams:
- **Zero-Setup Barrier**: Ready out of the box via single-command deployment (`docker compose up` or `rag-kit serve`).
- **No Component Hell**: Pre-wired with unified vector+sparse storage (Qdrant), universal LLM connectivity (LiteLLM), embedded local embeddings (FastEmbed), and built-in observability.
- **Dual-Mode Operation**:
  - **Local Mode**: 100% air-gapped, zero-egress operation using Ollama + FastEmbed ONNX on consumer CPUs/GPUs.
  - **Cloud Mode**: High-capability cloud synthesis using LiteLLM (OpenAI, Anthropic, Gemini) with zero local GPU requirements.
- **Enterprise Scale & Precision**: Handles from 1,000 to **10 Million+ documents** with memory-efficient indexing, structural table preservation, and citation attribution.

## Architecture Principles
1. **Spec & ADR First**: Every architectural fork (chunking strategy, vector DB selection, indexing mode, hybrid retrieval weighting, reranking pipeline) MUST be documented via an Architecture Decision Record (ADR) in `docs/decisions/` before lock-in.
2. **Build the Glue, Leverage Battle-Tested OSS**:
   - We **build** the differentiators: format-aware chunking, table topology preservation, exact/near deduplication, untrusted XML context sandboxing, prompt injection defense, and citation verification.
   - We **leverage** proven OSS primitives: Qdrant (vectors + sparse), LiteLLM (LLM gateway), FastEmbed (embeddings), and OpenTelemetry.
3. **Phase-by-Phase Modular Delivery**:
   - **Phase 1: Ingestion & Document Preprocessing Engine**: Multi-format document loading, format-aware chunking (PDF, Word, Markdown, Tables), metadata enrichment, exact/near-deduplication, and benchmark evaluations.
   - **Phase 2: Storage & Vector Indexing Layer**: Production Qdrant abstraction, dense HNSW + native sparse vector configuration, INT8 scalar quantization, and multi-tenant collection management.
   - **Phase 3: Hybrid Retrieval & Fusion**: Concurrent two-fold retrieval (Lexical BM25 Sparse + Dense Bi-Encoder) with Reciprocal Rank Fusion (RRF, $k=60$) and circuit breakers.
   - **Phase 4: Reranking & Quality Filter**: Cross-encoder reranking, threshold gating, context compression, and noise elimination.
   - **Phase 5: Synthesis & Generation Guardrails**: LiteLLM integration, context sandboxing (`<context_document>`), prompt injection prevention, and citation attribution.
   - **Phase 6: Turnkey REST API & Minimal Web UI**: FastAPI service, drag-and-drop document upload, and responsive citation-backed chat interface.
   - **Phase 7: Packaging & Production Hardening**: Single-command `docker compose up`, `.env` + `config.yaml` unified configuration, and health monitoring.
4. **Configuration Hierarchy**:
   - `.env`: Secrets, API keys, endpoints, and ports.
   - `config.yaml`: Version-controlled pipeline tuning parameters (chunk sizes, retrieval thresholds, weights, model selections).

## Engineering Disciplines & Skill Workflows
This repository leverages `addyosmani/agent-skills`:
- **ADRs**: `skills/documentation-and-adrs` for all architectural trade-offs in `docs/decisions/` (`ADR-001-...`).
- **Brainstorming & Ideation**: `skills/idea-refine` and `skills/interview-me` for vetting architectural trade-offs.
- **Specs & Task Planning**: `skills/spec-driven-development` and `skills/planning-and-task-breakdown`.
- **TDD & Code Health**: `skills/test-driven-development`, `skills/code-review-and-quality`, `skills/code-simplification`.
- **Performance & Observability**: `skills/performance-optimization`, `skills/observability-and-instrumentation`.
