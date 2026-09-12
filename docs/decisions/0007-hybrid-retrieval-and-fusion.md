# ADR-007: Concurrent Hybrid Retrieval, Reciprocal Rank Fusion, and Circuit Breaker Fallbacks

## Status
Accepted

## Date
2026-09-13

## Context
In enterprise knowledge search across 10M+ documents, relying on a single retrieval method creates a severe blind spot:
1. **The Semantic Blind Spot**: Dense bi-encoder embeddings excel at paraphrases and conceptual questions (e.g., *"How do I submit travel receipts?"*), but catastrophically fail on exact keywords, rare technical acronyms, part numbers, and error codes (e.g., `CVE-2024-3094`, `RFC-4122`, `TS-992`, `401(k)`). In enterprise benchmarks, dense-only retrieval misses 20–35% of relevant technical queries.
2. **The Lexical Blind Spot**: Keyword BM25 excels at exact token matches, but cannot understand semantic intent or synonyms (e.g. searching for *"automobile compensation"* will miss documents that mention *"car allowance"*).
3. **Score Calibration Conflict**: Cosine similarity is bounded $[-1, 1]$, whereas BM25 scores are unbounded $[0, \infty)$ and document-length sensitive. Combining them via linear score addition ($\alpha \cdot S_{dense} + (1-\alpha) \cdot S_{sparse}$) requires dynamic min-max normalization per query, which is unstable and brittle.
4. **Latency & Reliability (P95 Budget)**: If the dense vector store spikes in latency or suffers a network hiccup, the query must not fail. The system must degrade gracefully to sparse keyword search to maintain high availability.

## Decision

We implement a **Two-Fold Concurrent Hybrid Retrieval Engine** with **Reciprocal Rank Fusion (RRF)** and **Circuit Breaker Fallbacks**:

```
                              User Query
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
          [Dense Bi-Encoder]           [Sparse BM25 Index]
          - FastEmbed / Cloud          - Tokenized Lexical
          - Qdrant Dense HNSW          - Exact Match & Acronyms
          - Timeout: 150ms             - Timeout: 50ms
                    │                           │
                    └─────────────┬─────────────┘
                                  ▼
                     [Async Fan-Out Collector]
           (Degrades gracefully to Sparse if Dense times out)
                                  │
                                  ▼
                [Reciprocal Rank Fusion (RRF, k=60)]
                 RRF(d) = Σ 1 / (60 + rank_m(d))
                                  │
                                  ▼
              [Score Normalization & Deduplication Gate]
            (Filters duplicates, applies score_threshold)
                                  │
                                  ▼
                       Top-K Fused Candidates
```

### 1. Reciprocal Rank Fusion (RRF)
Instead of arbitrary linear score weighting, we fuse rankings using the industry-standard Reciprocal Rank Fusion formula:
$$RRF(d) = \sum_{m \in M} \frac{w_m}{k + rank_m(d)}$$
Where:
- $M = \{\text{dense}, \text{sparse}\}$
- $k = 60$ (smoothing constant empirically established by Cormack et al. to balance top-heavy and lower-tail rank positions)
- $w_{dense} = 0.6$ and $w_{sparse} = 0.4$ (configurable via `config.yaml`)
- Rank-invariant: does not depend on raw score magnitudes, eliminating the score calibration problem.

### 2. Concurrent Async Fan-Out & Circuit Breaker
- Ingestion and query fan-out uses `asyncio.gather()` to query Dense and Sparse indexes in parallel.
- **Circuit Breaker**: Dense vector search has a strict timeout budget ($150\text{ms}$). If dense search exceeds this deadline or fails with a connection error, the retriever catches the exception and returns the Sparse BM25 candidate list directly, logging a circuit-breaker trip event.

### 3. Clean Interface Protocol
We define `BaseHybridRetriever` in `recall.core.interfaces`:
```python
class BaseHybridRetriever(Protocol):
    async def retrieve(
        self,
        query: str,
        limit: int = 20,
        filter_dict: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[SearchResult]: ...
```

## Alternatives Considered

### 1. Linear Score Normalization ($\alpha S_{dense} + (1-\alpha) S_{bm25}$)
- *Pros*: Intuitive mental model.
- *Cons*: Requires per-query min-max normalization or global corpus statistics. If a single document has an anomalously high BM25 score, it compresses all other scores, destroying rank order.
- *Rejected*: RRF is rank-based, parameter-free, and consistently outperforms linear weighting across BEIR benchmarks.

### 2. Sequential / Cascade Retrieval (Sparse First, Dense Second)
- *Pros*: Slightly fewer vector computations on exact hits.
- *Cons*: Increases P95 latency (sequential waterfall instead of parallel fan-out). If the sparse stage has low recall, the dense stage never sees the candidates.
- *Rejected*: Parallel async fan-out achieves sub-100ms P95 latency.

## Consequences
- **Enterprise Precision**: Both exact error codes/acronyms and natural conversational queries are retrieved with high Recall@K and Precision@K.
- **Resilience**: A vector database slowdown never breaks the user experience; the service falls back gracefully to lexical results.
- **Clean Step to Phase 4**: Top-50 fused candidates from RRF feed directly into the Cross-Encoder Reranker in Phase 4.
