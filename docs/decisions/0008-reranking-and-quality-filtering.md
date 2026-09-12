# ADR-008: Cross-Encoder Reranking, Threshold Gating, and Context Compression

## Status
Accepted

## Date
2026-09-13

## Context
While the Phase 3 Concurrent Hybrid Retrieval engine achieves high recall across both semantic and lexical queries (typically Recall@50 > 95%), feeding raw top-20 or top-50 candidates directly into an LLM generation context introduces severe production failure modes:

1. **Context Bloat & Token Cost**: 20 chunks at 500 tokens each equals 10,000 tokens of input context. Across enterprise workloads with thousands of daily queries, this leads to excessive latency and high cloud API bills.
2. **"Lost in the Middle" Attention Failure**: Research (Liu et al., 2023) demonstrates that LLMs effectively attend to information at the very beginning and very end of prompt contexts, while critical evidence positioned in the middle suffers dramatic degradation in retrieval fidelity.
3. **Distractor Hallucinations**: Bi-encoder vector search often retrieves topically adjacent but factually irrelevant "distractor" passages. When presented to an LLM without quality filtering, distractors trigger ungrounded hallucinations.
4. **Bi-Encoder vs. Cross-Encoder Accuracy Gap**: Bi-encoders encode queries and documents into separate vector spaces independently, missing fine-grained token-level cross-interactions. A cross-encoder performs full all-to-all cross-attention across the $(query, document)$ pair simultaneously, achieving superior precision (MRR@10 improvements of +15% to +25%).

## Decision

We introduce **Phase 4: Reranking, Threshold Gating, and Context Compression**, structured as a three-stage precision pipeline:

```
                  Top-50 Candidates from Hybrid RRF
                                  │
                                  ▼
           ┌──────────────────────────────────────────────┐
           │        Stage 1: Cross-Encoder Reranker       │
           │  - Local: FlashRank ONNX (TinyBERT / MiniLM) │
           │  - Zero GPU / PyTorch, <30ms on CPU          │
           │  - Cloud: Optional Cohere / LiteLLM Rerank   │
           └──────────────────────┬───────────────────────┘
                                  │
                                  ▼
           ┌──────────────────────────────────────────────┐
           │        Stage 2: Score Threshold Gating       │
           │  - Calibrated cutoff (e.g. score >= 0.35)    │
           │  - Discards low-relevance distractor noise   │
           └──────────────────────┬───────────────────────┘
                                  │
                                  ▼
           ┌──────────────────────────────────────────────┐
           │        Stage 3: Extractive Compressor        │
           │  - Sentence-level saliency extraction        │
           │  - Compresses 500-token chunks to ~150 tok   │
           │  - Preserves exact source citation metadata  │
           └──────────────────────┬───────────────────────┘
                                  │
                                  ▼
                Top-K High-Precision Compressed Context
```

### 1. Zero-GPU Local Cross-Encoder via FlashRank
- For **Local Mode**, we integrate `flashrank` (ONNX runtime, quantized `ms-marco-TinyBERT-L-2-v2` or `ms-marco-MiniLM-L-12-v2`), requiring no PyTorch and having a footprint under 35MB.
- Reranks 20–50 candidates in 15–30ms on modern CPUs (Apple Silicon / Intel Xeon).
- Produces normalized relevance probabilities $[0, 1]$.

### 2. Strict Threshold Gating
- A configurable `min_rerank_score` (default: 0.35) eliminates distractor chunks before they enter the prompt.
- If no candidates exceed the threshold, the system triggers a fallback or signals that no trustworthy knowledge exists, preventing hallucinated answers.

### 3. Extractive Context Compressor
- Extracts only the most relevant sentences around query tokens and high-information segments.
- Bounded by `max_tokens_per_chunk` (default: 200 tokens) or `max_total_context_tokens` (default: 1500 tokens).
- Maintains strict provenance: every compressed snippet retains `doc_id`, `source_uri`, `chunk_id`, and `page_number`.

### 4. Interface Protocols
We define standard protocols in `recall.core.interfaces`:
```python
@runtime_checkable
class BaseReranker(Protocol):
    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int = 5,
        score_threshold: float | None = None,
    ) -> list[SearchResult]: ...

@runtime_checkable
class BaseContextCompressor(Protocol):
    def compress(
        self,
        query: str,
        candidates: list[SearchResult],
        max_tokens_per_chunk: int = 200,
        max_total_tokens: int = 1500,
    ) -> list[SearchResult]: ...
```

## Alternatives Considered

### 1. LLM-as-a-Reranker (Prompting GPT-4o / Claude to Rank Chunks)
- *Pros*: High semantic nuance.
- *Cons*: High latency (1.5–3.0s per rerank call) and high cost ($0.03–$0.08 per query). Defeats the zero-setup, local-first SMB mission.
- *Rejected*: Small ONNX cross-encoders achieve comparable ranking fidelity at 1/100th the latency and zero API cost.

### 2. Passing All Candidates Directly to LLM (No Reranking or Compression)
- *Pros*: Zero implementation overhead.
- *Cons*: Suffers from "Lost in the Middle", wastes 60–80% of prompt token budget on irrelevant background paragraphs, and increases hallucination rates.
- *Rejected*: Incompatible with enterprise-scale accuracy requirements.

## Consequences
- **Cost Reduction**: Shrinks prompt context tokens by 60%–80%, reducing cloud API costs proportionately.
- **Accuracy Improvement**: Reranking consistently lifts MRR and NDCG by 15–25% on enterprise technical benchmarks.
- **Portability**: The ONNX engine runs everywhere out of the box (Docker, macOS, Linux, Windows) with zero driver installation.
