# ADR-006: Unified Qdrant Vector Store, Dense/Sparse Storage, Scalar Quantization, and FastEmbed Integration

## Status
Accepted

## Date
2026-09-13

## Context
Phase 2 addresses storage and vector indexing for our turnkey enterprise RAG platform, **Recall**.
At scale (1,000 to 10M+ documents), typical RAG implementations suffer from two critical architectural pitfalls:
1. **Infrastructure Sprawl**: Deploying one database for dense embeddings (e.g., Pinecone/Milvus) and a separate Elasticsearch/Solr cluster for sparse keyword BM25 retrieval doubles operational overhead, RAM consumption, and network failure points.
2. **Memory Bloat**: Storing 10M 1536-dimensional float32 dense vectors requires ~60 GB of RAM just for raw vectors, plus additional overhead for HNSW graph edges. For SMBs and cost-conscious enterprise teams, this infrastructure cost is prohibitive.
3. **Setup Friction**: In local/dev environments, requiring developers to spin up external container clusters before running tests creates developer friction. We need zero-dependency, in-memory execution for testing alongside production Docker readiness.

## Decision

We adopt **Qdrant** as our single unified vector database engine and **FastEmbed** as our embedded local embedding provider:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          STORAGE & VECTOR INDEXING LAYER                    │
│                                                                             │
│  [Embedding Gateway]                                                        │
│  - Local: FastEmbed (ONNX, BAAI/bge-small-en-v1.5, 384-dim, zero API keys)  │
│  - Cloud: LiteLLM / OpenAI (text-embedding-3-small, 1536-dim)               │
│  - Sparse Lexical: FastEmbed SPLADE / BM25 token weights                    │
│         │                                                                   │
│         ▼                                                                   │
│  [BaseVectorStore Protocol (core/interfaces.py)]                            │
│         │                                                                   │
│         ▼                                                                   │
│  [QdrantVectorStore Implementation (storage/qdrant.py)]                     │
│  - Modes: In-Memory (":memory:") for tests | Remote (HTTP) for Production   │
│  - Unified Named Vectors:                                                   │
│      • "dense": HNSW Index (Cosine distance, m=16, ef_construct=100)        │
│      • "sparse": Native Sparse Vector index (BM25 token weights)            │
│  - INT8 Scalar Quantization: 75% RAM reduction with <1% recall trade-off    │
│  - Payload Indexing: Fast filtering on doc_id, file_type, policy, page_num  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1. Unified Dense + Sparse Storage in Qdrant
Instead of running separate vector and keyword engines, Qdrant natively supports multiple named vectors per point:
- **Dense Vector (`"dense"`)**: Bi-encoder semantic representation.
- **Sparse Vector (`"sparse"`)**: Token-frequency / BM25 lexical weights.
This allows our upcoming Phase 3 (Hybrid Retrieval) to query dense and sparse vectors in a single database round-trip with native score blending or reciprocal rank fusion.

### 2. INT8 Scalar Quantization (SQ)
For enterprise scale:
- 32-bit floating point vectors are quantized to 8-bit integers (`int8`).
- Memory footprint drops from 4 bytes/dim to 1 byte/dim (75% savings).
- Vectors are kept in RAM as int8 for fast HNSW traversal; original vectors can be stored on-disk for precision rescoring.

### 3. Dual-Mode Connection (Zero-Setup vs. Docker)
The `QdrantVectorStore` abstracts connection details via configuration:
- **Testing & Local Dev**: Initializes `QdrantClient(location=":memory:")` or local SQLite directory. No Docker container or network port required.
- **Production Deployment**: Connects to `http://qdrant:6333` (defined in `docker-compose.yml` and `.env`).

### 4. FastEmbed Embedded ONNX Runtime
- Runs locally via `onnxruntime` without pulling heavy PyTorch dependencies (~2GB saved in container image).
- Pre-downloads optimized models like `BAAI/bge-small-en-v1.5` (384 dims, fast CPU throughput: ~1,500 sentences/sec).

## Alternatives Considered

### 1. ChromaDB
- *Pros*: Simple local Python API.
- *Cons*: Weak native sparse vector support; limited production clustering; uncalibrated scalar quantization at 10M scale.
- *Rejected*: Inadequate for 10M enterprise scale with native sparse hybrid search.

### 2. Milvus
- *Pros*: Robust distributed clustering.
- *Cons*: Heavy dependency footprint (etcd, MinIO, Pulsar required even for modest setups).
- *Rejected*: Violates our North Star of zero component sprawl and simple single-container deployment for SMBs.

### 3. Pinecone / Cloud-Only Vector DBs
- *Pros*: Fully managed.
- *Cons*: Violates our local, air-gapped dual-mode promise. Recurring egress and vector storage SaaS bills.
- *Rejected*: Must support 100% self-hosted local mode out of the box.

## Consequences
- **Memory Efficiency**: 10M documents can be indexed in under 12GB RAM using INT8 scalar quantization.
- **Developer Velocity**: In-memory mode enables unit and integration tests to run in milliseconds in CI without external services.
- **Hybrid Readiness**: Seamlessly paves the way for Phase 3 Hybrid Retrieval without schema migration.
