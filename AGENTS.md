# Project Guidelines: Production Enterprise RAG Kit

## Mission & North Star
We are building a **turnkey, production-grade, enterprise-scale Retrieval-Augmented Generation (RAG) platform & toolkit**.
Designed for open-source adoption by small-to-medium businesses (SMBs) and enterprise engineering teams:
- **Zero-Setup Barrier**: Ready out of the box via single-command deployment (`docker compose up` or `rag-kit serve`).
- **No Component Hell**: Pre-wired with unified vector+sparse storage (Qdrant), universal LLM connectivity (LiteLLM), embedded local embeddings (FastEmbed), and built-in observability.
- **Dual-Mode Operation**:
  - **Local Mode**: 100% air-gapped, zero-egress operation using Ollama + FastEmbed ONNX on consumer CPUs/GPUs.
  - **Cloud Mode**: High-capability cloud synthesis using LiteLLM (OpenAI, Anthropic, Gemini) with zero local GPU requirements.
- **Enterprise-Grade Retrieval**: Hybrid dense+sparse search, table preservation, dedup, reranking, and citations. **Validated today** on real BEIR benchmarks up to **10,000 documents** (FiQA@10k, Sep 2026). Hyperscale targets (100k–10M+) are on the roadmap — see [README Status & Roadmap](README.md#status--roadmap) and [ADR-012](docs/decisions/0012-synthetic-corpus-and-scale-benchmarking.md) (synthetic streaming generator not yet built). Do not claim 10M+ production readiness without published stress reports.

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

This repository uses **[Craft](https://github.com/DTiapan/craft)** (`using-craft`) to route into **[addyosmani/agent-skills](https://github.com/addyosmani/agent-skills)** on the agent machine (reference-only, not vendored in git). See [CONSTRAINTS.md](CONSTRAINTS.md) and [docs/craft-setup.md](docs/craft-setup.md) for the binding contract.

- **Ledger first:** Read `docs/engineering-ledger/INDEX.md` before non-trivial work.
- **One phase, one skill:** Route via `using-craft`; do not stack process skills.
- **Mandatory Skill Dispatch:** Apply the workflow skill for the active phase (`documentation-and-adrs`, `spec-driven-development`, `test-driven-development`, etc.).
- **Zero Drift Mandate**:
  - No premature implementation: Stay within the active phase. Do not build downstream layers until upstream foundations are tested green.
  - Spec & ADR First: Any architectural fork or new subsystem requires an ADR in `docs/decisions/` before code lock-in.
  - TDD Verification: All code changes must have tests. 100% of tests must pass on `.venv` before commit.
  - Quality Bar Preservation: Never skip tests, weaken assertions, or bypass errors to achieve green status.

## Craft (orchestration + ledger)

- Non-trivial work: read `docs/engineering-ledger/INDEX.md` first.
- Route phases via `using-craft` skill (reference-only — do not copy Addy or Craft skills into this repo).
- Append DR/LL/INDEX before ending substantive sessions; tag lessons with `Scope: project | universal`.
- Irreversible forks: ADR in `docs/decisions/` per `documentation-and-adrs`.
- New machine setup: [docs/craft-setup.md](docs/craft-setup.md)
- Project manifest: [craft.project.yaml](craft.project.yaml)

