# Recall: The Open-Source, Turnkey Enterprise RAG Platform

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Architecture: ADRs](https://img.shields.io/badge/architecture-13%20ADRs%20recorded-blue.svg)](docs/decisions/)
[![Tests](https://img.shields.io/badge/tests-91%20passed-brightgreen.svg)]()
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

> **Recall** is a turnkey, open-source Retrieval-Augmented Generation (RAG) platform that deploys in one click with zero setup—providing self-hosted hybrid search, table-aware structural chunking, cross-encoder reranking, and dual-mode local (Ollama) and cloud (LiteLLM) synthesis for enterprise knowledge bases scaling from 1,000 to 10M+ documents.

---

## Key Highlights

- **Zero-Setup Barrier**: Ready out of the box via single-command deployment (`docker compose up` or `recall serve`). No component sprawl or glue scripts required.
- **Embedded Web UI**: Out-of-the-box modern dark-mode chat interface with drag-and-drop file ingestion and clickable, page-specific citation popovers.
- **Dual-Mode Operation**:
  - **Local Mode**: 100% air-gapped, zero-data-leakage execution using [FastEmbed](https://github.com/qdrant/fastembed) (ONNX CPU), [FlashRank](https://github.com/PrithivirajDamodaran/FlashRank) local cross-encoders, and [Ollama](https://github.com/ollama/ollama) (Llama 3.2 / Mistral).
  - **Cloud Mode**: High-capability cloud synthesis using [LiteLLM](https://github.com/BerriAI/litellm) (OpenAI, Anthropic, Gemini, Cohere) with zero local GPU requirements.
- **Enterprise Scale & Precision**:
  - **Structural Format Adapters**: Native ingestion for PDF, Microsoft Word (`.docx`), Markdown, and plain text.
  - **Table Topology Preservation**: Markdown table serialization that retains column headers across sub-chunks without severing numbers.
  - **Deduplication Gate**: 64-bit `xxhash` exact deduplication (<100MB RAM for 10M docs) + MinHash LSH for near-duplicate filtering.
  - **Concurrent Hybrid Retrieval**: Dense HNSW vector similarity in Qdrant fused with BM25+ lexical search via Reciprocal Rank Fusion (RRF, $k=60$) with circuit breaker fallbacks.
  - **Cross-Encoder Reranking & Compression**: Zero-GPU ONNX cross-encoders with extractive sentence reduction to eliminate prompt bloat and prevent "Lost in the Middle" attention failures.
  - **Context Sandboxing & Citation Verification**: Untrusted document passages sandboxed in XML tags (`<context_document>`) with strict inline citation verification (`[Doc X, p. Y]`).

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          RECALL PLATFORM ARCHITECTURE                       │
│                                                                             │
│  [Embedded Web UI]             [FastAPI REST Engine]         [Recall CLI]   │
│  - Drag & Drop Ingest          - POST /v1/ingest             - recall serve │
│  - Verified Citation Pills     - POST /v1/search             - recall ingest│
│  - Real-time Diagnostics       - POST /v1/chat               - recall query │
│         │                               │                           │       │
│         └───────────────────────┬───────┴───────────────────────────┘       │
│                                 ▼                                           │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                     RECALL CORE ORCHESTRATION PIPELINE                │  │
│  │                                                                       │  │
│  │  1. Format Ingestion Engine (PDF, DOCX, Markdown, Table Formatter)    │  │
│  │  2. Exact & Near Deduplication (xxhash + MinHash LSH)                 │  │
│  │  3. Concurrent Hybrid Retrieval (Dense HNSW + BM25+ Sparse)           │  │
│  │  4. Reciprocal Rank Fusion (RRF, k=60) with Dense Circuit Breakers    │  │
│  │  5. Cross-Encoder Reranking (FlashRank ONNX) + Threshold Gating       │  │
│  │  6. Extractive Context Compression (Saliency Extractor)               │  │
│  │  7. XML Context Sandboxing & Prompt Injection Defense                 │  │
│  │  8. LiteLLM Universal Synthesis + Grounded Citation Verification      │  │
│  └───────────────────┬───────────────────────────────┬───────────────────┘  │
│                      │                               │                      │
│                      ▼                               ▼                      │
│  ┌─────────────────────────────────┐   ┌─────────────────────────────────┐  │
│  │      STORAGE & VECTOR ENGINE    │   │        LLM & EMBEDDING GATEWAY  │  │
│  │      (Qdrant Unified OSS)       │   │        (LiteLLM + FastEmbed)    │  │
│  │  - Dense HNSW Index             │   │  [Local Mode]:                  │  │
│  │  - Native Sparse BM25+ Index    │   │  - FastEmbed ONNX (bge-small)   │  │
│  │  - Scalar Quantization (INT8)   │   │  - FlashRank ONNX Reranker      │  │
│  │  - Multi-tenant Collections     │   │  - Ollama (Llama-3 / Mistral)   │  │
│  │                                 │   │  [Cloud Mode]:                  │  │
│  │                                 │   │  - OpenAI, Claude 3.5, Gemini   │  │
│  └─────────────────────────────────┘   └─────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Quickstart

### Option A: Turnkey Single-Command Docker Deployment (Recommended)

```bash
# Clone the repository
git clone https://github.com/DTiapan/Recall.git
cd Recall

# Launch complete stack (FastAPI + Qdrant + Web UI)
docker compose up -d

# Open the Web UI in your browser
open http://localhost:8000
```

### Option B: Local Python CLI

```bash
# Clone and enter directory
git clone https://github.com/DTiapan/Recall.git
cd Recall

# Create virtual environment and install
uv venv
source .venv/bin/activate
uv pip install -e ".[formats]"

# Copy environment configuration
cp .env.example .env

# Start the server and embedded Web UI
recall serve
```

---

## Command-Line Interface (CLI)

```bash
# Start the API and UI server
recall serve --host 0.0.0.0 --port 8000

# Ingest any document into the active collection
recall ingest path/to/document.pdf --collection documents

# Query the pipeline directly from the terminal
recall query "What are the production access requirements?"

# Run retrieval benchmarks on real-world datasets
recall benchmark --dataset sample
uv pip install -e ".[benchmark]"
RAG_MODE=local recall benchmark --dataset beir:scifact --limit 500
```

---

## Retrieval Benchmarks

Recall evaluates hybrid retrieval on **real-world corpora** with human relevance labels — not synthetic templates. See [ADR-013](docs/decisions/0013-real-world-benchmark-datasets.md).

**Full benchmark narrative** (expected results, improvements, scale roadmap to 10M+): [docs/benchmarks/README.md](docs/benchmarks/README.md).

**Environment:** `RAG_MODE=local`, FastEmbed `BAAI/bge-small-en-v1.5` (dense + sparse), in-memory Qdrant, Apple Silicon CPU (Sep 2026).

| Dataset | Docs | Queries | HitRate@5 | MRR | Rerank HitRate@5 | Rerank MRR |
|---|---:|---:|---:|---:|---:|---:|
| [Bundled sample](data/sample/) (MD, DOCX, PDF) | 5 | 10 | **100.0%** | **0.875** | **100.0%** | **1.000** |
| [BEIR SciFact](https://github.com/beir-cellar/beir) | 500 | 35 | **85.7%** | **0.757** | — | — |
| [BEIR FiQA](https://github.com/beir-cellar/beir) @10k | 10,000 | 243 | **57.6%** | **0.463** | **51.4%** | **0.416** |

Sample benchmark also reports query P50 **40 ms** and rerank P50 **60 ms** on local FastEmbed (Sep 2026).

FiQA @10k scale run (`batch_size=128`, in-memory Qdrant, Apple Silicon CPU, Sep 2026): ingest **6.7 docs/sec** (~25 min), peak RSS **15.3 GB**, query P50 **257 ms**, P95 **299 ms**, P99 **324 ms**, rerank P50 **280 ms**. Full report: [docs/benchmarks/fiqa-10k.md](docs/benchmarks/fiqa-10k.md).

```bash
# Bundled enterprise corpus (no extra deps, no network)
RAG_MODE=local QDRANT_URL=:memory: recall benchmark --dataset sample

# Standard IR benchmark (downloads ~2.7 MB on first run)
uv pip install -e ".[benchmark]"
RAG_MODE=local QDRANT_URL=:memory: recall benchmark --dataset beir:scifact --limit 500

# Scale tier (batched embed + bulk upsert, batch_size=128)
RAG_MODE=local QDRANT_URL=:memory: recall benchmark --dataset beir:fiqa --scale 10k --batch-size 128

# High-throughput scale stress (100k–10M index/latency, mock embeddings — not IR quality)
RAG_MODE=local QDRANT_URL=:memory: recall benchmark --dataset beir:fiqa --scale 100k --fast
```

---

## REST API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | `GET` | Serves the interactive zero-dependency dark-mode Web UI |
| `/v1/health` | `GET` | Healthcheck returning vector store connectivity and active dimensions |
| `/v1/stats` | `GET` | Indexed document count and storage metrics |
| `/v1/ingest` | `POST` | Upload document files (PDF, DOCX, MD, TXT) or raw text payloads |
| `/v1/search` | `POST` | Concurrent hybrid search (Dense HNSW + BM25+) with RRF fusion |
| `/v1/chat` | `POST` | End-to-end RAG synthesis returning verified citation audit trails |

---

## Architecture Decision Records (ADRs)

Every architectural milestone in Recall is documented prior to implementation:

| ADR | Title | Status |
|---|---|---|
| [ADR-001](docs/decisions/0001-record-architecture-decisions.md) | Record Architecture Decisions | Accepted |
| [ADR-002](docs/decisions/0002-production-rag-architecture.md) | Production RAG Architecture and Technology Stack | Accepted |
| [ADR-003](docs/decisions/0003-ingestion-and-chunking-strategy.md) | Ingestion and Structural Chunking Strategy | Accepted |
| [ADR-004](docs/decisions/0004-format-aware-chunking-adapters.md) | Format-Aware Chunking Adapters and Table Serialization | Accepted |
| [ADR-005](docs/decisions/0005-turnkey-enterprise-rag-platform.md) | Turnkey Enterprise RAG Platform Positioning and Brand | Accepted |
| [ADR-006](docs/decisions/0006-vector-storage-and-indexing.md) | Production Qdrant Storage, INT8 Quantization, and FastEmbed | Accepted |
| [ADR-007](docs/decisions/0007-hybrid-retrieval-and-fusion.md) | Concurrent Hybrid Retrieval, Reciprocal Rank Fusion, and Circuit Breakers | Accepted |
| [ADR-008](docs/decisions/0008-reranking-and-quality-filtering.md) | Cross-Encoder Reranking, Threshold Gating, and Context Compression | Accepted |
| [ADR-009](docs/decisions/0009-synthesis-and-guardrails.md) | Context Sandboxing, Injection Defense, LiteLLM, and Citation Verification | Accepted |
| [ADR-010](docs/decisions/0010-turnkey-rest-api-and-web-ui.md) | Turnkey REST API, Embedded Web UI, and CLI Tooling | Accepted |
| [ADR-011](docs/decisions/0011-packaging-and-production-hardening.md) | Multi-Stage Hardened Dockerfile and Compose Orchestration | Accepted |
| [ADR-012](docs/decisions/0012-synthetic-corpus-and-scale-benchmarking.md) | Synthetic Enterprise Corpus Generation and Multi-Scale Stress Benchmarking | Accepted |
| [ADR-013](docs/decisions/0013-real-world-benchmark-datasets.md) | Real-World Benchmark Datasets (BEIR + Curated Corpus) | Accepted |

---

## Running the Test Suite

All changes are governed by strict test-driven development:

```bash
# Fast parallel unit/integration suite (smoke tests excluded by default)
pytest

# Docker deployment smoke test (requires Docker)
pytest -m smoke -n 0 -v

# Export traces to an OTLP collector (optional)
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
```

---

## Built with open tooling

Recall is the **product** in a three-repo stack for serious GenAI engineering:

| Project | Role |
|---------|------|
| **Recall** (this repo) | Production RAG: ingest, hybrid retrieval, real BEIR evals |
| [Craft](https://github.com/DTiapan/craft) | Workflow router + [engineering ledger](docs/engineering-ledger/INDEX.md) (decisions, lessons, phase gates) |
| [Battery](https://github.com/DTiapan/battery) | Local MCP agent memory (hybrid search, `BATTERY.md` in git) |

Craft is adopted in this repo (`docs/engineering-ledger/`, [setup guide](docs/craft-setup.md)). Battery complements agent session memory while Recall handles document retrieval.

---

## Author

**Ajas Bakran** — AI systems engineer focused on agent evaluation, context engineering, and production reliability.

- GitHub: [github.com/DTiapan](https://github.com/DTiapan)
- LinkedIn: [linkedin.com/in/ajasbakran](https://linkedin.com/in/ajasbakran)
- Newsletter: [growithai.substack.com](https://growithai.substack.com/)

Advisory and consulting on AI agent reliability, memory architectures, MCP integrations, and production RAG — [get in touch](mailto:bakran.ajas@gmail.com).

---

## License

MIT License. See [LICENSE](LICENSE) for details.
