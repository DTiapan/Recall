# Retrieval Benchmark — beir:fiqa@10k (AP-003 post-fix)

**Date:** 2026-09-13  
**Hardware:** Apple Silicon Mac, CPU-only FastEmbed ONNX  
**Command:**

```bash
PYTHONUNBUFFERED=1 RAG_MODE=local recall benchmark \
  --dataset beir:fiqa --scale 10k --batch-size 128 --rerank --subsample-seed 42 \
  --output docs/benchmarks/fiqa-10k-ap003-rerank.md
```

**Context:** AP-003 Phase A validation after rerank fixes (threshold split, `candidate_k=50`, MiniLM-L-12 wired from config). Fair subsample covers **648/648** evaluable queries (was 243 with first-N bias). Index persisted on disk for reuse.

**Compared to pre-fix run:** [fiqa-10k.md](fiqa-10k.md) — 57.6% hybrid / 51.4% rerank on 243 queries; rerank regression was the threshold bug, not FiQA difficulty.

---

- Documents ingested: **10000**
- Queries evaluated: **648**
- HitRate@5 (hybrid retrieval): **73.8%**
- MRR (hybrid retrieval): **0.614**
- nDCG@10 (pytrec_eval): **0.522**
- Recall@10: **0.593**
- Recall@100: **0.853**
- MRR@10 (recip_rank): **0.614**
- Subsample: **10000 docs**, seed **42**, **648** evaluable queries in corpus
- Persisted index (built): **`~/.cache/recall/benchmark-indexes/beir-fiqa_10k_s42`**
- HitRate@5 (rerank, pool=50): **72.1%**
- MRR (rerank): **0.593**
- Rerank latency P50: **1607.5ms**
- Ingest time: **671.47s**
- Ingest throughput: **14.9 docs/sec**
- Peak RSS: **14835 MB**
- Query latency P50: **247.2ms**
- Query latency P95: **291.1ms**
- Query latency P99: **349.6ms**

**Verdict:** Threshold regression eliminated. Rerank −1.7pp vs hybrid is within expected dataset-specific variance (see LL-005); no longer a correctness bug.
