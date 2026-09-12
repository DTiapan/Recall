# Recall: The Open-Source, Turnkey Enterprise RAG Platform

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Architecture: ADRs](https://img.shields.io/badge/architecture-ADRs%20recorded-blue.svg)](docs/decisions/)
[![Tests](https://img.shields.io/badge/tests-29%20passed-brightgreen.svg)]()
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

> **Recall** is a turnkey, open-source Retrieval-Augmented Generation (RAG) platform that deploys in one click with zero setup—providing self-hosted hybrid search, table-aware structural chunking, and dual-mode local (Ollama) and cloud (LiteLLM) synthesis for enterprise knowledge bases scaling from 1,000 to 10M+ documents.

---

## Key Highlights

- **Zero-Setup Barrier**: Single-command deployment (`docker compose up` or `recall serve`). No complex orchestration or component sprawl.
- **Dual-Mode Operation**:
  - **Local Mode**: 100% air-gapped, zero-data-leakage execution using [FastEmbed](https://github.com/qdrant/fastembed) (ONNX CPU/GPU) and [Ollama](https://github.com/ollama/ollama) (Llama-3/Mistral).
  - **Cloud Mode**: High-capability cloud synthesis using [LiteLLM](https://github.com/BerriAI/litellm) (OpenAI, Anthropic, Gemini) with zero local GPU requirements.
- **Enterprise Precision**:
  - **Structural Format Adapters**: Native ingestion for PDF, Microsoft Word (`.docx`), Markdown, and plain text.
  - **Table Topology Preservation**: Markdown table serialization that retains column headers across sub-chunks without severing numbers.
  - **Deduplication Gate**: 64-bit `xxhash` exact deduplication (<100MB RAM for 10M docs) + MinHash LSH for near-duplicate filtering.
  - **Citation Attribution**: Full provenance tracking with exact page numbers, section heading breadcrumbs, and document timestamps.
  - **Prompt Injection Defense**: Untrusted context sandboxed inside bounded XML tags (`<context_document>`).

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          RECALL PLATFORM ARCHITECTURE                       │
│                                                                             │
│  [Minimal Web UI]              [REST API (FastAPI)]         [CLI Tooling]   │
│  - Drag & Drop Ingest          - /v1/ingest                 - recall serve  │
│  - Cited Chat Stream           - /v1/query                  - recall ingest │
│  - Collection Manager          - /v1/health                                 │
│         │                               │                           │       │
│         └───────────────────────┬───────┴───────────────────────────┘       │
│                                 ▼                                           │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                     RECALL CORE ORCHESTRATION ENGINE                  │  │
│  │  - Unified Configuration Engine (.env + config.yaml)                  │  │
│  │  - Structural Ingestion Engine (Adapters, Table Formatter, Dedup)     │  │
│  │  - Hybrid Search & Fusion Engine (RRF, Circuit Breakers)              │  │
│  │  - Context Sandboxing & Citation Attribution Guardrails               │  │
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
│  │                                 │   │    Gemini, Azure, Bedrock)      │  │
│  └─────────────────────────────────┘   └─────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Quickstart

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/your-org/recall.git
cd recall

# Create virtual environment and install with format support
uv venv
source .venv/bin/activate
uv pip install -e ".[dev,formats]"
```

### 2. Configuration

Recall uses a two-tier configuration hierarchy:
- **`.env`**: Secrets, API keys, ports, and deployment mode (`RAG_MODE=cloud` or `local`).
- **`config.yaml`**: Pipeline tuning parameters (chunk size, overlap, hybrid weights, HNSW graph parameters).

```bash
cp .env.example .env
# Edit .env with your LLM API keys (or leave default for local mode)
```

### 3. Programmatic Usage

```python
from pathlib import Path
from recall.adapters import ChunkingAdapterRegistry
from recall.core.config import load_config
from recall.core.models import IngestConfig

# Load unified configuration (.env + config.yaml)
config = load_config()

# Initialize format-aware adapter registry
registry = ChunkingAdapterRegistry()

# Ingest and structure any document (PDF, Word, Markdown, Text)
chunks = registry.process(
    file_path=Path("quarterly_financial_report.pdf"),
    config=IngestConfig(chunk_size=500, chunk_overlap=50),
)

for chunk in chunks:
    print(f"[{chunk.metadata.content_type.upper()}] Page {chunk.metadata.page_number} | Tokens: {chunk.metadata.token_count}")
    print(chunk.contextualized_text[:120], "...\n")
```

---

## Running Tests

All changes are governed by strict test-driven development:

```bash
pytest tests/ -v
```

---

## Engineering Discipline & Standards

- **Architecture Decision Records (ADRs)**: All architectural forks are documented in [`docs/decisions/`](docs/decisions/).
- **Binding Constraints**: Quality floors and zero-drift mandates are recorded in [`CONSTRAINTS.md`](CONSTRAINTS.md).
- **Agent Skill Workflows**: Enforced via [`AGENTS.md`](AGENTS.md) utilizing `addyosmani/agent-skills`.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
