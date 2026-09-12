# ADR-010: Turnkey FastAPI REST Interface, Embedded Citation Chat UI, and CLI

## Status
Accepted

## Date
2026-09-13

## Context
A production enterprise RAG engine cannot require custom glue code for every deployment or integration. To achieve the North Star objective—**Zero-Setup Barrier & No Component Hell for SMBs and Engineering Teams**—the system requires:

1. **Standardized REST API**: A uniform OpenAPI-documented HTTP interface for ingestion, hybrid search, and citation-backed synthesis usable by web apps, microservices, and external agents.
2. **Interactive Zero-Dependency Web UI**: An out-of-the-box UI served directly by FastAPI without needing Node.js, npm, or complex frontend build steps, allowing instant verification via drag-and-drop document upload and citation-backed conversation.
3. **Single-Command CLI (`recall serve`)**: A unified entrypoint that parses `.env` and `config.yaml`, initializes Qdrant, FastEmbed, BM25, FlashRank, and LiteLLM, and launches the server.

## Decision

We implement **Phase 6: Turnkey REST API & Embedded Web UI**:

```
                       User / Browser / Microservice
                                     │
                                     ▼
                ┌─────────────────────────────────────────┐
                │          FastAPI Service (:8000)        │
                │  - /v1/ingest (multipart file upload)   │
                │  - /v1/search (dense + sparse hybrid)   │
                │  - /v1/chat   (end-to-end RAG + cite)   │
                │  - /v1/health (system diagnostics)      │
                │  - / (Embedded Vanilla Dark UI)         │
                └────────────────────┬────────────────────┘
                                     │
                                     ▼
                      Unified RAG Engine Container
       (QdrantVectorStore, BM25Index, FlashRank, LiteLLM)
```

### 1. API Endpoints
- `POST /v1/ingest`: Accepts `multipart/form-data` (PDF, DOCX, Markdown, TXT, JSON) or raw text. Routes through `ChunkingAdapterRegistry`, dedup cleaner, generates FastEmbed vectors, and updates Qdrant & BM25 indexes simultaneously.
- `POST /v1/search`: Accepts `{query, limit, filter_dict}`. Dispatches concurrent hybrid retrieval and returns top RRF fused chunks.
- `POST /v1/chat`: Accepts `{query, collection, stream}`. Runs full pipeline (Hybrid Retrieve -> FlashRank Rerank -> Context Compress -> LiteLLM Synthesize -> Citation Verifier), returning structured answer and verified citation breadcrumbs.
- `GET /v1/health`: Checks vector store health, embedding model dimension, and active collections.
- `GET /`: Serves a modern, zero-dependency dark-mode HTML5/CSS3/Vanilla JS application featuring drag-and-drop file upload, real-time citation bubbles, and responsive streaming cards.

### 2. Embedded Web UI Design System
In accordance with production design guidelines:
- Dark slate background (`#0b0f19`), electric indigo accents (`#6366f1`), frosted glassmorphism borders (`rgba(255,255,255,0.08)`), and modern Inter typography.
- Clickable citation tags `[Doc 1, p. 3]` that pop open an inspectable provenance modal showing the exact chunk text, source URI, and confidence score.
- 100% self-contained in static assets with zero external CDN dependency risks.

### 3. CLI Command
`recall serve --host 0.0.0.0 --port 8000 --reload` loads configuration and boots Uvicorn.

## Alternatives Considered
- Separate Next.js / React frontend repository: Creates component sprawl, requires `node`/`npm` installation, breaking the single-command zero-setup barrier.
- Gradio / Streamlit: Rigid layout, heavy dependency footprint (>200MB of extra wheels), clunky mobile experience, and difficult to customize for enterprise embedding.
- FastHTML: Emerging, but pure HTML/CSS/JS served directly via FastAPI is universally compatible, zero-install, and lighter.

## Consequences
- Single command `recall serve` gives users an immediate, fully functional enterprise search and chat platform.
- Zero extra runtime requirements: Python + pip/uv only.
- Microservices can consume the REST API directly while business users use the Web UI.
