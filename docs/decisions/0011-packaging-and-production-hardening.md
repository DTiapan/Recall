# ADR-011: Packaging, Docker Multi-Stage Containerization, and Single-Command Orchestration

## Status
Accepted

## Date
2026-09-13

## Context
To fulfill Recall's North Star—**Zero-Setup Barrier & No Component Hell**—the platform must deploy reliably in any enterprise environment (bare-metal, local developer machines, Kubernetes, air-gapped VPCs) with a single command. 

Requirements:
1. **Single-Command Launch**: `docker compose up` must launch a complete, self-healing stack (FastAPI server, Qdrant vector store, embedded Web UI, and optional Ollama) with zero manual network wiring or prerequisite installations.
2. **Hardened Multi-Stage Container**: The `Dockerfile` must use multi-stage builds to exclude compiler toolchains, run as an unprivileged non-root user (`recall:10001`), and minimize attack surface.
3. **Configuration Hierarchy**:
   - `.env`: Secrets, API keys, endpoints, and ports.
   - `config.yaml`: Version-controlled pipeline tuning parameters.
4. **Health Monitoring & Probes**: Liveness and readiness endpoints (`/v1/health`) for Docker healthchecks and Kubernetes probes.

## Decision

We introduce **Phase 7: Packaging & Production Hardening**:

### 1. Hardened Dockerfile
- Base image: `python:3.11-slim-bookworm`
- Non-root user: `recall` (UID 10001)
- Explicit security attributes: read-only filesystem where possible, `/tmp` mounted with tmpfs, explicit `HEALTHCHECK` probe against `/v1/health`.
- Multi-stage build separates build-time dependencies from the lean production runtime image.

### 2. Turnkey Docker Compose Architecture
- Services:
  - `recall`: The primary API and Web UI service, binding port `8000:8000`.
  - `qdrant`: Official `qdrant/qdrant:v1.9.2` image with persistent vector volume storage.
  - `ollama` (optional profile `local`): In-cluster Ollama container for air-gapped zero-egress local inference.
- Internal network `recall-network` isolates database communication.

### 3. Production Configuration
- `config.yaml`: Pre-configured with balanced production defaults (384-dim BGE embeddings, BM25+ sparse retrieval, RRF k=60, FlashRank reranker, and citation sandboxing).
- `.env.example`: Documented variable catalog covering local mode and cloud API credentials.

## Consequences
- Developers and SMBs can run `docker compose up -d` and have an enterprise RAG platform running within seconds.
- Fully compatible with cloud container services (AWS ECS, Google Cloud Run, Azure Container Instances, Kubernetes).
- Clean security posture satisfying enterprise SOC2/ISO container audit standards.
