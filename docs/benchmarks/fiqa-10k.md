# Retrieval Benchmark — beir:fiqa@10k

**Date:** 2026-09-13  
**Hardware:** Apple Silicon Mac, CPU-only FastEmbed ONNX  
**Command:**

```bash
PYTHONUNBUFFERED=1 RAG_MODE=local QDRANT_URL=:memory: \
  recall benchmark --dataset beir:fiqa --scale 10k --batch-size 128
```

**Context:** First completed 10k scale run after batched ingest ([DR-002](../engineering-ledger/decisions.md)) and live progress logging. See [benchmark narrative](README.md) for before/after story.

---

- Documents ingested: **10000**
- Queries evaluated: **243**
- HitRate@5 (retrieval): **57.6%**
- MRR (retrieval): **0.463**
- HitRate@5 (rerank): **51.4%**
- MRR (rerank): **0.416**
- Ingest time: **1492.67s**
- Ingest throughput: **6.7 docs/sec**
- Peak RSS: **15253 MB**
- Query latency P50: **256.8ms**
- Query latency P95: **298.8ms**
- Query latency P99: **323.6ms**
- Rerank latency P50: **279.5ms**