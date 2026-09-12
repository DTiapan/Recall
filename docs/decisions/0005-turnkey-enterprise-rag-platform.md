# ADR-005: Turnkey Enterprise Architecture, Dual-Mode Engine, and OSS Leverage Stack

## Status
Accepted

## Date
2026-09-13

## Context
Small-to-medium enterprises (SMBs) and engineering teams face substantial friction when adopting RAG:
1. **Fragmented Component Hell**: Deploying typical RAG stacks requires stitching together vector databases, sparse search engines, embedding services, LLM gateways, observability collectors, and custom web frontends across dozens of disjointed config files.
2. **Setup Friction**: If setup requires extensive manual configuration or specialized ML expertise, adoption plummets. The system must operate "out of the box" via a single command (`docker compose up` or `rag-kit serve`).
3. **Deployment Duality (Local vs. Cloud)**: Some organizations require 100% air-gapped, zero-data-leakage local execution (Ollama + local embeddings on CPU/GPU), while others demand top-tier cloud LLM synthesis (OpenAI, Anthropic, Gemini) with zero local GPU requirement.
4. **Configuration Ergonomics**: Critical environment variables (secrets, endpoints, ports) should reside in `.env`, while pipeline tuning parameters (chunk sizes, retrieval thresholds, weights) belong in a clean, versioned `config.yaml`.

## Decision

We establish the **Turnkey Enterprise Platform Architecture**, operating as a cohesive, single-command system built around strategic open-source leverage:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ONE-CLICK TURNKEY PLATFORM                         │
│                                                                             │
│  [Web UI (Minimal v1)]         [REST API (FastAPI)]         [CLI Tooling]   │
│  - Drag & Drop Ingest          - /v1/ingest                 - rag-kit serve │
│  - Chat + Citations            - /v1/query                  - rag-kit ingest│
│  - Collection Status           - /v1/health                                 │
│         │                               │                           │       │
│         └───────────────────────┬───────┴───────────────────────────┘       │
│                                 ▼                                           │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                    CORE ORCHESTRATION LAYER (OUR CODE)                │  │
│  │  - Unified Configuration Engine (.env + config.yaml)                  │  │
│  │  - Structural Ingestion Engine (Adapters, Table Formatter, Dedup)     │  │
│  │  - Hybrid Search & Fusion Engine (RRF, Circuit Breakers)              │  │
│  │  - Context Assembly & Citation Attribution Guardrails                 │  │
│  └───────────────────┬───────────────────────────────┬───────────────────┘  │
│                      │                               │                      │
│                      ▼                               ▼                      │
│  ┌─────────────────────────────────┐   ┌─────────────────────────────────┐  │
│  │      STORAGE & VECTOR ENGINE    │   │        LLM & EMBEDDING GATEWAY  │  │
│  │      (Qdrant Unified OSS)       │   │        (LiteLLM + FastEmbed)    │  │
│  │  - Dense HNSW Index             │   │  [Local Mode]:                  │  │
│  │  - Native Sparse (BM25) Vector  │   │  - FastEmbed (ONNX CPU/GPU)     │  │
│  │  - Scalar Quantization (INT8)   │   │  - Ollama (Llama-3/Mistral)     │  │
│  │  - Payload Filtering & Tenancy  │   │  [Cloud Mode]:                  │  │
│  │                                 │   │  - LiteLLM (OpenAI, Anthropic,  │  │
│  │                                 │   │    Gemini, Bedrock, Azure)      │  │
│  └─────────────────────────────────┘   └─────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1. The Build vs. Leverage Strategy
We build what differentiates the quality of enterprise answers and leverage battle-tested open-source primitives for commodity infrastructure:
- **Build (Our IP)**:
  - Format-aware structural chunking (Markdown, PDF, Word, Tables) preserving 2D headers and breadcrumb hierarchies.
  - Near-deduplication (MinHash LSH) and exact deduplication (xxhash).
  - Hybrid reciprocal rank fusion (RRF) with circuit breakers and fallback policies.
  - Untrusted XML context sandboxing, prompt injection defense, and citation verification.
  - Unified FastAPI server, CLI, and minimal chat/upload Web UI.
- **Leverage (OSS Giants)**:
  - **Vector & Sparse Engine**: **Qdrant** (single binary / container). Provides native dense vectors + native sparse vectors in one engine, eliminating the need to run a separate Elasticsearch/Solr cluster.
  - **LLM Gateway**: **LiteLLM**. Unified interface supporting 100+ LLM providers with automatic retries, rate-limiting, and standard OpenAI-compatible completions.
  - **Local Embedding Engine**: **FastEmbed** (by Qdrant). Lightweight ONNX-based runtime with no heavy PyTorch dependencies; runs out of the box on consumer CPUs.
  - **Observability**: **OpenTelemetry** + **structlog**. Out-of-the-box structured JSON logs and standard distributed trace headers.

### 2. Dual-Mode Deployment: Local vs. Cloud
The platform supports two deployment profiles configured via `.env`:

| Mode | Embeddings | LLM Provider | Hardware Requirement | Data Privacy |
|---|---|---|---|---|
| **Local Mode** | FastEmbed (`BAAI/bge-small-en-v1.5`) | Ollama (`llama3:8b` / `mistral`) | Consumer CPU / Apple Silicon | 100% Air-gapped, zero external egress |
| **Cloud Mode** | Cloud or FastEmbed (`text-embedding-3-small`) | LiteLLM (`gpt-4o`, `claude-3-5-sonnet`, `gemini-1.5-pro`) | Minimal (runs on standard VM / container) | Enterprise cloud with API keys |

### 3. Unified Configuration Hierarchy
- **`.env`**: Secrets and environment specifics (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `QDRANT_URL`, `PORT`, `RAG_ENV`). Never committed to source control.
- **`config.yaml`**: Pipeline parameters (chunk size, overlap, top_k, RRF constant, score threshold, rerank toggles). Version-controlled with sane defaults.
- Configuration is strongly typed and validated at application startup using Pydantic Settings.

### 4. Minimal Web UI for v1
The platform ships with a self-contained, responsive Web UI embedded directly in the service:
- Document upload pane (drag & drop PDF, Word, Markdown, Text).
- Ingestion status and document collection manager.
- Chat interface with stream response and expandable citation cards showing exact page numbers and source breadcrumbs.

## Consequences
- **Zero-Setup Barrier**: An enterprise can run `docker compose up` and immediately start querying corporate documents via UI and REST API.
- **Maintainability**: Replacing or upgrading LLM providers or embedding models requires editing a single line in `config.yaml` without changing pipeline code.
- **Portability**: Runs locally on a developer MacBook, on an air-gapped on-prem server, or in AWS/GCP Kubernetes clusters.
