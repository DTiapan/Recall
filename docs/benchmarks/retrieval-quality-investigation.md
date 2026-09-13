# Retrieval Quality Investigation — Research & Findings

> **Status:** Phase A + B complete (2026-09-13) — Rerank fixes validated; fair subsample + pytrec_eval + disk indexes; Phase C tuning open  
> **Goal:** Understand what is broken vs expected, borrow proven practices, and build a reusable knowledge base for Recall and future projects.  
> **Trigger:** FiQA@10k hybrid HitRate@5 **57.6%** (usable) but rerank **51.4%** (regression); SciFact **85.7%** at 500 docs vs FiQA at 10k.

---

## Executive summary

| Category | Verdict | Action |
|----------|---------|--------|
| SciFact 0% → 85.7% | **Fixed** (label bug) | Done — DR-001 |
| Batched ingest / progress | **Fixed** | Done — DR-002 |
| Rerank regression on FiQA | **Likely implementation bug + design gap** | AP-003 Phase A |
| 57.6% hybrid on FiQA@10k | **Partly expected, partly methodology** | AP-003 Phase B |
| 15.3 GB RSS at 10k | **Engineering debt** | AP-004 (scale), not quality |
| Rerank always-on in production | **May hurt some domains** | Research-backed config change |

**Key insight from external research:** Reranking is **not a free win**. Rigorous FiQA evals show cross-encoders can **help FiQA** but **hurt SciFact/NFCorpus** depending on setup ([vaibhavkev/rag-eval](https://github.com/vaibhavkev/rag-eval)). Our FiQA regression is still suspicious because we apply **absolute score thresholding only on the rerank path** — a known anti-pattern in production RAG.

---

## What the literature says (borrowed practices)

### 1. BEIR / FiQA baselines (are we “bad”?)

Published reference points on **full FiQA** (~57k docs, standard metrics):

| Method | nDCG@10 (approx.) | Source |
|--------|------------------:|--------|
| BM25 | 0.24 | BEIR paper, rag-eval |
| Dense `bge-base-en-v1.5` | 0.41 | rag-eval |
| Hybrid (dense + BM25, α≈0.7) | 0.42–0.43 | rag-eval, BEIR hybrid |
| + cross-encoder rerank (N=20) | 0.43–0.46 | rag-eval |
| Fine-tuned `bge-small` on FiQA | 0.39–0.45 | HuggingFace model cards |

Our **57.6% HitRate@5** on a **10k subsample** with `bge-small` is **not directly comparable** to nDCG@10 on full FiQA — but it is in the right ballpark for an untuned zero-shot stack, not catastrophically broken.

**Borrow:** Report **nDCG@10, Recall@10, MRR@10** via `pytrec_eval` alongside HitRate@5 so GitHub numbers match BEIR conventions.

### 2. Hybrid fusion

- BEIR authors: dense+sparse hybrid beats single-channel on most datasets ([arXiv:2306.07471](https://arxiv.org/html/2306.07471v1)).
- rag-eval on FiQA: hybrid beats dense-only by **+0.013 nDCG@10** (small but significant).
- **Recall uses RRF** (k=60, weights 0.6/0.4) — sound architecture, weights not tuned per dataset.

**Borrow:** Validate hybrid > dense-only on our stack; consider min-max fusion or adaptive α as a tuning step (not blocking).

### 3. Rerank candidate pool size

- BEIR official example: retrieve **top-100**, rerank, evaluate @10 ([beir evaluate_bm25_ce_reranking.py](https://github.com/beir-cellar/beir/blob/main/examples/retrieval/evaluation/reranking/evaluate_bm25_ce_reranking.py)).
- rag-eval on FiQA: **N=20 beat N=50 and N=100** on both quality and latency — deeper pools added reranker error.

**Our benchmark:** `retrieve_limit=20`, `search_limit=5` — pool size is reasonable; **not the primary bug**.

### 4. Absolute rerank thresholds are dangerous

Multiple production write-ups warn against fixed cutoffs (e.g. 0.35):

- Cross-encoder scores are **not calibrated probabilities**; they shift per query, domain, and model version ([DEV: score calibration](https://dev.to/ji_ai/cross-encoder-reranker-score-calibration-why-05-cutoffs-fail-1k9b)).
- Fixed thresholds create **asymmetric abstain rates** across query types.
- **Better approaches:** relative margin (top − second), z-score within candidate list, anchor-passage normalization, or **no threshold in eval**.

**Our code:** `search()` does **not** apply `score_threshold`; `search_rerank()` applies `retrieval.score_threshold: 0.35` to cross-encoder scores. This **asymmetrically penalizes rerank** in benchmarks.

### 5. Reranker model choice

| Model | Role | Notes |
|-------|------|-------|
| FlashRank `ms-marco-TinyBERT-L-2-v2` | **What we run** (~4MB, MS MARCO web search) | [FlashRank](https://github.com/PrithivirajDamodaran/FlashRank) |
| `BAAI/bge-reranker-base` | **What config.yaml says** | Not wired; different model family |
| `bge-reranker-v2-m3` | Strong OSS reranker on FiQA | BGE docs: +0.04 nDCG@10 vs large on FiQA |

**Borrow:** Align config → implementation. For finance IR, MS MARCO TinyBERT is a weak default; BGE reranker family is the documented OSS choice ([BGE reranker tutorial](https://bge-model.com/tutorial/5_Reranking/5.3.html)).

### 6. Reranking is dataset-dependent

rag-eval transfer experiment:

| Dataset | Hybrid | + rerank N=20 | Δ nDCG@10 |
|---------|-------:|--------------:|----------:|
| FiQA | 0.4347 | 0.4605 | **+0.026** |
| NFCorpus | 0.3823 | 0.3705 | −0.012 |
| SciFact | 0.7589 | 0.7501 | −0.009 |

**Borrow:** Make rerank **optional per collection or dataset**; never assume rerank helps everywhere. Measure on each target domain.

---

## What we found in Recall (code + benchmark evidence)

### Issue A — Config / implementation drift (confidence: **high**)

| Artifact | Says | Code does |
|----------|------|-----------|
| `config.yaml` `reranking.local_model` | `BAAI/bge-reranker-base` | Ignored |
| ADR-008 | FlashRank TinyBERT / MiniLM | `FlashRankReranker()` hardcoded to `ms-marco-TinyBERT-L-2-v2` |
| `RerankingSettings` in `config.py` | `local_model` field exists | Never passed to `FlashRankReranker` |

**Impact:** Operators think they run BGE reranker; benchmarks run MS MARCO TinyBERT.

### Issue B — Score threshold on rerank only (confidence: **high**)

```python
# service.py — hybrid path: NO threshold
return await retriever.retrieve(query, limit=limit, ...)

# service.py — rerank path: threshold 0.35 applied
reranked = self.reranker.rerank(..., score_threshold=score_thresh)
```

**Impact:** Relevant docs scoring 0.20–0.34 on cross-encoder are **dropped from rerank results** but counted in hybrid top-5. Explains **6.2 pp HitRate@5 drop** (57.6% → 51.4%).

**Fix hypothesis:** Disable threshold for eval; use relative margin or per-domain calibration for production.

### Issue C — Benchmark methodology gaps (confidence: **high**)

| Gap | Our behavior | Standard practice |
|-----|--------------|-------------------|
| Metrics | HitRate@5, MRR (custom) | nDCG@10, Recall@10, MAP@10 (pytrec_eval) |
| Corpus slice | ~~First N docs from dict iteration~~ → **qrels-coverage + seeded fill** (`--subsample-seed`) | Full corpus or reproducible subsample |
| Chunking | 1 chunk / BEIR document | Chunk long posts; BEIR text can exceed embed limits |
| SciFact rerank | Not reported | Measure both paths on all datasets |
| Eval isolation | Hybrid and rerank are separate `search()` calls | Same candidate pool for fair A/B |

### Issue D — Scale effects vs bugs (confidence: **medium**)

FiQA@10k vs SciFact@500 drop is **partly expected**:

- 20× more documents → more distractors in top-k.
- FiQA = informal finance Q&A; SciFact = scientific claims (different difficulty).
- `bge-small` zero-shot on FiQA: published nDCG@10 ~0.39 ([fine-tuned FiQA retriever cards](https://huggingface.co/vivekkopthsd/fiqa-retriever-bge-small)).

**Not a bug by itself** — but we should establish **FiQA@500** baseline (same methodology as SciFact) before attributing gaps to scale.

### Issue E — Memory at 10k (confidence: **high**, separate track)

- All chunks materialized in Python before ingest.
- In-memory Qdrant rebuild every run.
- 15.3 GB peak RSS — blocks 100k+ **real** embed on laptop.

**Not a retrieval-quality bug** — blocks honest scale stress with real embeddings.

---

## Fixes applied (Phase A — 2026-09-13)

### Fix 1 — Rerank threshold no longer steals from hybrid ✅

**Before:** `search_rerank()` used `retrieval.score_threshold: 0.35` (hybrid had no threshold).  
**After:** New `reranking.score_threshold` defaults to **`null`** (rank-only). Production can opt in after calibration.

**Files:** `config.yaml`, `core/config.py`, `api/service.py`

### Fix 2 — Config drives reranker model ✅

**Before:** Hardcoded `ms-marco-TinyBERT-L-2-v2`; config said `bge-reranker-base` (ignored).  
**After:** `reranking.local_model` → `resolve_flashrank_model_name()` → FlashRank ONNX. Default upgraded to **`ms-marco-MiniLM-L-12-v2`**. BGE HF names map to MiniLM until a native BGE CrossEncoder path exists.

**Files:** `config.yaml`, `rerank/flashrank.py`, `api/service.py`

### Validation runs (post-fix)

| Dataset | Hybrid HR@5 | Rerank HR@5 | Verdict |
|---------|------------:|-------------:|---------|
| SciFact@500 | **85.7%** | 82.9% | Small rerank dip — matches literature (rerank not universal on SciFact) |
| FiQA@500 | **88.2%** | **88.2%** | **Flat** — threshold bug was the FiQA@10k regression driver |
| FiQA@10k (pre-fix) | 57.6% | 51.4% | 243 queries, first-N subsample — **superseded** |
| FiQA@10k (post-fix) | **73.8%** | **72.1%** | 648 queries, fair subsample — threshold bug fixed; rerank −1.7pp (expected) |

Reports: [scifact-500-ap003.md](scifact-500-ap003.md), [fiqa-500-ap003.md](fiqa-500-ap003.md), [fiqa-10k-ap003-rerank.md](fiqa-10k-ap003-rerank.md)

---

## Prioritized fix plan (AP-003)

### Phase A — Correctness

- [x] Wire `reranking.local_model` from config
- [x] Separate `reranking.score_threshold` (default null)
- [x] Re-run SciFact@500 + FiQA@500
- [x] Add `reranking.candidate_k` (default 50) wired through service, query(), and benchmark
- [x] Fix benchmark rerank metric aggregation (was always reporting 0%)
- [x] CLI `--rerank` and `--rerank-pool` flags
- [x] Re-run FiQA@10k — hybrid **73.8%** HR@5, rerank **72.1%** (648 queries, fair subsample); [report](fiqa-10k-ap003-rerank.md)
- [ ] Optional: dataset-specific rerank enable flag (SciFact may skip rerank)

**Done when:** ~~FiQA@10k rerank ≥ hybrid~~ — rerank −1.7pp documented as dataset-specific (LL-005); threshold regression eliminated.

### Fix 3 — Disk-persisted benchmark indexes ✅

**Before:** `QDRANT_URL=:memory:` in docs; partial Qdrant dirs without `manifest.json` could not reuse.  
**After:** Default slot at `~/.cache/recall/benchmark-indexes/<slug>/` stores `qdrant/`, `manifest.json`, `subsample.json`, and `reports/latest_*.md`. Incomplete slots auto-rebuild.

**Files:** `eval/benchmark_index.py`, `cli.py`, `retrieval_benchmark.py`

### Phase B — Fair measurement (credibility for GitHub)

1. [x] Add **pytrec_eval** metrics (nDCG@10, Recall@10/100, MRR@10) to benchmark report — hybrid retrieval only by default (`include_rerank=False`).
2. [x] **Qrels-aware seeded subsample** for `--limit` / `--scale` (greedy query-coverage + seeded fill, not first-N dict order). CLI: `--subsample-seed` (default 42).
3. [x] **SciFact@500** fair subsample — **300** evaluable queries (was **35**); nDCG@10 **0.860**, HR@5 **91.0%** ([report](scifact-500-fair.md)).
4. [x] **FiQA@500** fair subsample — **500/500** evaluable queries (was **17/648**); nDCG@10 **0.722**, HR@5 **78.8%** ([report](fiqa-500-fair.md)).
5. [x] Re-run **FiQA@10k** with fair subsample — **648/648** queries; hybrid **73.8%**, rerank **72.1%**, nDCG@10 **0.522** ([report](fiqa-10k-ap003-rerank.md)).

### Phase C — Quality tuning (iterative, exploratory)

1. Try `ms-marco-MiniLM-L-12-v2` vs TinyBERT (FlashRank supported).
2. Evaluate BGE reranker (may need sentence-transformers, not FlashRank).
3. Tune hybrid weights on FiQA dev split (small grid).
4. Chunk long FiQA documents (>512 tokens) with ingest pipeline.
5. Consider domain-specific embedder fine-tune (longer horizon).

### Phase D — Scale (after A + B green)

See [benchmarks README](README.md) — `--fast` stress tiers, streaming ingest, cloud VM.

---

## Universal lessons (portable to other projects)

| ID | Lesson | Scope |
|----|--------|-------|
| LL-001 | Eval label keys must match indexed metadata | universal |
| LL-004 | Never apply absolute cross-encoder thresholds without per-query calibration | universal |
| LL-005 | Rerank is dataset-dependent — measure on transfer sets before enabling globally | universal |
| LL-006 | Report the same metrics as your benchmark community (BEIR → nDCG@10) or comparisons mislead | universal |

---

## References

- [BEIR paper](https://arxiv.org/abs/2104.08663) — benchmark design
- [BEIR hybrid baselines](https://arxiv.org/html/2306.07471v1) — dense+sparse fusion
- [vaibhavkev/rag-eval](https://github.com/vaibhavkev/rag-eval) — hand-rolled FiQA eval, rerank N=20 finding, transfer degradation
- [Cross-encoder threshold calibration](https://dev.to/ji_ai/cross-encoder-reranker-score-calibration-why-05-cutoffs-fail-1k9b)
- [BGE reranker evaluation on FiQA](https://bge-model.com/tutorial/5_Reranking/5.3.html)
- [FlashRank models](https://github.com/PrithivirajDamodaran/FlashRank)
- Recall: [ADR-007](../decisions/0007-hybrid-retrieval-and-fusion.md), [ADR-008](../decisions/0008-reranking-and-quality-filtering.md), [ADR-013](../decisions/0013-real-world-benchmark-datasets.md)
