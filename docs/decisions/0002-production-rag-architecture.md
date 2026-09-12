# ADR-002: Production RAG End-to-End Architecture, Query Transformation, and Retrieval Fusion

## Status
Accepted

## Date
2026-09-13

## Context
We are engineering an enterprise-grade, open-source Retrieval-Augmented Generation (RAG) framework capable of scaling up to 10+ million documents. Production deployments require rigorous adherence to performance (P95 latency budgets), high retrieval recall and precision, resilience against system faults (graceful degradation), strict security (protecting against direct and indirect prompt injections), and query adaptability (handling complex, ambiguous, or multi-part enterprise queries).

To avoid brittle, monolithic pipelines, the system must be decoupled into testable, plug-and-play components where each module can be substituted or tuned independently.

## Decision

We adopt a modular, component-by-component pipeline architecture structured into four decoupled layers:

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                        RUNTIME RETRIEVAL & SYNTHESIS FLOW                     │
│                                                                               │
│  User Query                                                                   │
│      │                                                                        │
│      ▼                                                                        │
│  [1. Query Transformation & Sanitization]                                     │
│      ├─ Injection & Jailbreak Defense (Safety Filter)                         │
│      ├─ Query Rewriting & Expansion (Synonyms, Domain Acronyms)               │
│      └─ Query Decomposition (Multi-query fan-out for complex questions)       │
│      │                                                                        │
│      ▼                                                                        │
│  [2. Two-Fold Hybrid Retrieval with Circuit Breakers]                         │
│      ├─ Parallel Fan-Out: Dense Bi-Encoder + Sparse BM25                      │
│      ├─ Fault Tolerance: Degrade to Sparse if Vector DB times out (>150ms)    │
│      └─ Fusion: Reciprocal Rank Fusion (RRF, k=60)                            │
│      │                                                                        │
│      ▼                                                                        │
│  [3. Quality Filtering & Cross-Encoder Reranking]                             │
│      ├─ Rerank top-50 candidates via Cross-Encoder                            │
│      ├─ Fallback: Return top RRF scores if reranker latency exceeds budget    │
│      └─ Deduplication & Context Compression (eliminate redundant chunks)      │
│      │                                                                        │
│      ▼                                                                        │
│  [4. Secure Context Assembly & Attributed Synthesis]                          │
│      ├─ Document Sandboxing (Explicit untrusted XML boundary markers)         │
│      ├─ System Prompt Role Guardrails (Prevent hallucination & domain drift)  │
│      └─ Citation Attribution Gate (Verify claims against source chunk IDs)    │
└───────────────────────────────────────────────────────────────────────────────┘
```

### 1. Query Transformation & Sanitization Layer
- **Input Sanitization**: Detects adversarial jailbreaks, command injection patterns, and abnormal unicode characters before touching retrieval indexes.
- **Query Rewriting & Disambiguation**: Resolves conversational coreferences (e.g. "What about its operating margin?" &rarr; "What about Apple's Q3 2023 operating margin?").
- **Multi-Query Decomposition**: Decomposes complex multi-faceted inquiries into atomic subqueries:
  - *Example*: *"Compare cloud revenue growth of Microsoft and Google in Q4"* &rarr; Subquery A: *"Microsoft cloud revenue growth Q4"*, Subquery B: *"Google cloud revenue growth Q4"*.
  - Subqueries execute retrieval in parallel; candidate pools are deduplicated and merged via RRF.

### 2. Hybrid Retrieval with Graceful Fallbacks
- **Dual Indexing**:
  - **Dense Vectors**: HNSW index with Scalar Quantization (INT8) or IVF-PQ to support 10M vectors within predictable RAM limits (<8 GB for 768-dim embeddings).
  - **Sparse Lexical**: BM25 / SPLADE index for exact term matches, part numbers, and error codes.
- **Asynchronous Concurrent Fan-Out**: Query both indexes via `asyncio.gather()` with a strict 150ms deadline.
- **Circuit Breaker / Fallback**: If the vector database spikes or fails, fallback seamlessly to BM25 sparse results alone to guarantee high service availability.
- **Reciprocal Rank Fusion (RRF)**:
  $$RRF(d) = \sum_{m \in M} \frac{1}{k + rank_m(d)} \quad (k = 60)$$
  Avoids uncalibrated linear score merging between disparate score distributions.

### 3. Reranking & Quality Filter
- **Two-Stage Retrieval**: Pull top 50–100 candidates from hybrid fusion; rerank down to top 10–20 using a lightweight Cross-Encoder (e.g., `bge-reranker-base` or Cohere Rerank API).
- Reduces top-20 retrieval failure rate by up to 67% over naive dense retrieval.
- **Timeout Protection**: If reranking latency exceeds 250ms, return the top-K from RRF directly.

### 4. Security, Red Teaming & Role Adherence
- **Indirect Prompt Injection Defense**: Documents retrieved from corporate intranets, PDFs, or external web pages cannot be trusted. Chunks are strictly encapsulated in bounded XML tags (`<context_document id="..."> ... </context_document>`). The system prompt instructs the LLM that content inside `<context_document>` is untrusted reference data and must never be interpreted as instructions.
- **Role Guardrails**: System prompts enforce domain boundaries; attempts to steer the agent outside its knowledge domain trigger explicit refusal responses.
- **Attribution & Hallucination Defense**: The synthesis engine validates that cited source IDs exist in the retrieved chunk set and flags ungrounded statements.

## Alternatives Considered

### 1. Pure Dense Semantic Retrieval (Bi-Encoder Only)
- *Pros*: Simple pipeline, single index to maintain.
- *Cons*: High failure rate on rare identifiers, technical serial numbers, error codes (e.g. `TS-999`), and exact phrases. Misses 20–35% of relevant documents in enterprise search benchmarks.
- *Rejected*: Enterprise RAG demands both semantic understanding and exact keyword precision.

### 2. Linear Score Weighting ($\alpha \cdot S_{dense} + (1-\alpha) \cdot S_{sparse}$)
- *Pros*: Intuitive weighting knob.
- *Cons*: BM25 scores are unbounded $[0, \infty)$ and document-length sensitive, while cosine similarity is $[-1, 1]$. Normalizing both requires computing corpus statistics or min-max scaling per query, leading to calibration instability.
- *Rejected*: RRF ($k=60$) is parameter-free, rank-invariant, and empirically superior in multi-modal information retrieval.

### 3. All-in-One Monolithic Frameworks (LangChain / LlamaIndex Black-Box Chains)
- *Pros*: Quick prototyping.
- *Cons*: Difficult to profile latency at microsecond level, hard to plug custom circuit breakers, opaque prompt injection vectors, high dependency churn.
- *Rejected*: Building clean, minimal, protocol-based interfaces (using Pydantic v2 and standard async Python) ensures zero bloat and true open-source modularity.

## Consequences
- **Engineering Discipline**: Each phase delivers an independently runnable, benchmarked module before proceeding to the next.
- **Latency Budget**: Sub-350ms P95 target is achievable through parallel async IO, scalar vector quantization, and circuit breakers.
- **Operational Scalability**: Modular database abstractions make it trivial to swap between in-memory testing (DuckDB / Faiss), single-node deployments (Qdrant), and distributed clusters (Milvus) at 10M scale.
