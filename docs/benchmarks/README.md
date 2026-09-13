# Recall Benchmark Results & Scale Roadmap

> **Purpose:** Credible, reproducible evidence for GitHub and enterprise evaluators — what we measured, what we fixed, and how we plan to stress-test to **10M+ documents**.

**Last updated:** 2026-09-13  
**Environment:** Apple Silicon Mac, `RAG_MODE=local`, FastEmbed `BAAI/bge-small-en-v1.5` (dense + sparse), disk-backed Qdrant (`~/.cache/recall/benchmark-indexes/`)

---

## Results at a glance

| Dataset | Docs | Queries | HitRate@5 | nDCG@10 | Recall@10 | MRR | Notes |
|---------|-----:|--------:|----------:|--------:|----------:|----:|-------|
| [Bundled sample](../../data/sample/) | 5 | 10 | **100.0%** | — | — | **0.875** | MD, DOCX, PDF — format-aware chunking |
| [BEIR SciFact](https://github.com/beir-cellar/beir) @500 | 500 | **300** | **91.0%** | **0.860** | **0.935** | **0.839** | Fair subsample ([report](scifact-500-fair.md)) |
| [BEIR FiQA](https://github.com/beir-cellar/beir) @500 | 500 | **500** | **78.8%** | **0.722** | **0.854** | **0.686** | Fair subsample ([report](fiqa-500-fair.md)) |
| [BEIR FiQA](https://github.com/beir-cellar/beir) @10k | 10,000 | **648** | **73.8%** | **0.522** | **0.593** | **0.614** | Fair subsample + disk index ([report](fiqa-10k-ap003-rerank.md)); rerank 72.1% |

**Subsample methodology (Sep 2026):** `--limit` / `--scale` use qrels-aware greedy coverage + seeded fill (`--subsample-seed 42`), not first-N dict order. BEIR metrics via `pytrec_eval` (nDCG@10, Recall@10/100). Hybrid retrieval only by default.

**Persisted indexes (Sep 2026):** By default, `recall benchmark` writes a disk-backed Qdrant index to `~/.cache/recall/benchmark-indexes/<dataset>_<limit>_s<seed>/` with a `manifest.json` fingerprint. Re-running the same command **skips re-embed/re-ingest** and goes straight to query eval (~minutes → ~seconds for ingest). Use `--force-reindex` to rebuild, `--in-memory` for ephemeral CI-style runs, `--no-reuse-index` to re-ingest without deleting the slot.

### FiQA@10k scale metrics (Sep 2026)

| Metric | Value |
|--------|------:|
| Ingest throughput | **6.7 docs/sec** |
| Total ingest time | **~25 min** (1,493 s) |
| Peak RSS | **15.3 GB** |
| Query latency P50 / P95 / P99 | **257 / 299 / 324 ms** |
| Rerank latency P50 | **~280 ms** (hybrid-only) / **~1608 ms** (with `--rerank`, pool=50) |

---

## What we expected vs what we improved

Recall benchmarks use **real-world corpora with human relevance labels** ([ADR-013](../decisions/0013-real-world-benchmark-datasets.md)), not synthetic templates. We hit three problems on the way to a credible 10k scale run; each has a documented fix.

### 1. SciFact returned 0% HitRate@5 (label mismatch)

| | Before | After |
|---|--------|-------|
| **Symptom** | 0% HitRate@5 despite plausible-looking retrieval | **85.7%** HitRate@5, **0.757** MRR |
| **Root cause** | BEIR `doc_id` was replaced by a convenience hash at ingest; qrels could not join | — |
| **Fix** | Preserve `doc_id` through ingest + `relevant_sources` fallback ([DR-001](../engineering-ledger/decisions.md), [LL-001](../engineering-ledger/lessons.md)) | commit `5e0f296` |

**Lesson:** Eval label keys must match indexed metadata — never substitute convenience hashes.

### 2. 10k FiQA ingest was too slow and opaque

| | Before | After |
|---|--------|-------|
| **Symptom** | Per-document ONNX dispatch + Qdrant round-trips; run abandoned mid-flight with no visible progress | Completed 10k docs in **~25 min** with live 3-phase terminal progress |
| **Root cause** | `batch_size=1` ingest path ([LL-002](../engineering-ledger/lessons.md)) | — |
| **Fix** | Batched embed + upsert at `batch_size=128` ([DR-002](../engineering-ledger/decisions.md)); progress logging per batch | commit `2ea19cb`, AP-002 |

**Lesson:** Scale benchmarks need batched embed+upsert; long runs need explicit phase/batch progress on stdout.

### 3. FiQA@10k quality — acceptable for scale proof, not tuned for finance IR

| | Expected | Observed |
|---|----------|----------|
| **Retrieval** | Moderate drop vs smaller corpora as candidate noise grows | **57.6%** HR@5 — plausible for 10k finance Q&A with `bge-small` on CPU |
| **Rerank** | Flat or improvement over hybrid retrieval | **72.1%** HR@5 (−1.7pp vs hybrid) — threshold bug fixed; small dip is dataset-specific |
| **Memory** | Bounded growth with batching | **15.3 GB** peak RSS at 10k — too high for laptop 100k+ with real embeddings |

**Open follow-up:** See [retrieval-quality-investigation.md](retrieval-quality-investigation.md) — research-backed root cause analysis and AP-003 fix plan.

---

## How to reproduce

```bash
# Setup (once per machine)
uv venv && source .venv/bin/activate
uv pip install -e ".[formats,benchmark]"

# Quality benchmarks (real embeddings)
PYTHONUNBUFFERED=1 RAG_MODE=local QDRANT_URL=:memory: recall benchmark --dataset sample
PYTHONUNBUFFERED=1 RAG_MODE=local QDRANT_URL=:memory: recall benchmark --dataset beir:scifact --limit 500 --subsample-seed 42
PYTHONUNBUFFERED=1 RAG_MODE=local QDRANT_URL=:memory: recall benchmark --dataset beir:fiqa --limit 500 --subsample-seed 42

# Scale tier — real embeddings, batched ingest (expect ~25 min for 10k on Apple Silicon CPU)
PYTHONUNBUFFERED=1 RAG_MODE=local QDRANT_URL=:memory: \
  recall benchmark --dataset beir:fiqa --scale 10k --batch-size 128 \
  --output docs/benchmarks/fiqa-10k.md
```

Use `PYTHONUNBUFFERED=1` so phase/batch progress prints immediately.

---

## Scale stress test roadmap (10k → 10M+)

**Goal:** Prove the pipeline holds from SMB (10k) to hyperscale (10M+) on throughput, memory, and latency — without pretending a laptop can run 10M real ONNX embeddings overnight.

### Two benchmark tiers (by design)

| Tier | Mode | Embeddings | Measures | Laptop feasible? |
|------|------|------------|----------|------------------|
| **Quality** | default | FastEmbed ONNX (real) | HitRate@5, MRR, rerank | Up to **~10k–50k** docs |
| **Stress** | `--fast` | Mock (deterministic) | Ingest docs/sec, RSS, P50/P95/P99 latency | **100k – 10M** index shape |

`--fast` is intentional ([DR-002](../engineering-ledger/decisions.md)): it stress-tests **indexing, Qdrant, hybrid search, and latency percentiles** — not IR quality. See [ADR-012](../decisions/0012-synthetic-corpus-and-scale-benchmarking.md).

```bash
# Stress tier examples (mock embeddings — index/latency only)
PYTHONUNBUFFERED=1 RAG_MODE=local QDRANT_URL=:memory: \
  recall benchmark --dataset beir:fiqa --scale 100k --fast --batch-size 128

PYTHONUNBUFFERED=1 RAG_MODE=local QDRANT_URL=:memory: \
  recall benchmark --dataset beir:fiqa --scale 1m --fast --batch-size 512
```

### Recommended ladder

| Step | Scale | Mode | Where to run | Purpose |
|------|------:|------|--------------|---------|
| ✅ Done | 10k | Real | Local Mac | Quality + batched ingest baseline |
| **Next** | 100k | `--fast` | Local Mac or CI large runner | Validate index/latency curve; estimate 1M cost |
| **Then** | 1M | `--fast` | Cloud VM (32–64 GB RAM) | RAM + disk footprint; Qdrant INT8 quantization |
| **Target** | 10M | `--fast` | Cloud VM / bare metal (128+ GB RAM) | Hyperscale soak; needle-query latency at depth |

### Why 10k real took ~25 minutes on this CPU

Bottlenecks today:

1. **FastEmbed ONNX on CPU** — batched, but still ~5–7 chunks/sec at `batch_size=128`.
2. ~~**In-memory Qdrant** — rebuild every run.~~ **Disk-backed benchmark indexes** (default) with manifest reuse; `--in-memory` for ephemeral runs.
3. ~~**All chunks materialized in Python** before ingest~~ **Streaming micro-batch ingest** — chunk → embed → upsert without holding the full corpus in RAM.

Rough projection **without code changes**:

| Scale | Real embed (this Mac) | `--fast` (this Mac) |
|------:|----------------------:|--------------------:|
| 10k | **~25 min** (measured) | ~2–5 min (estimate) |
| 100k | **~4+ hours** | ~15–30 min (estimate) |
| 1M | **Days** — impractical | Cloud VM, ~2–4 hours (estimate) |
| 10M | **Not viable locally** | Cloud VM 128 GB+, streaming ingest required |

### Ways to go faster

| Approach | Speedup | Quality impact | Best for |
|----------|---------|----------------|----------|
| **`--fast` mock embeddings** | 10–50× ingest | None for IR metrics | 100k–10M stress |
| **Larger `batch_size`** (256–512) | Modest | None | GPU or high-RAM hosts |
| **GPU embeddings** (CUDA/CoreML ONNX) | 5–20× | None | Real embed at 100k+ |
| **Persistent Qdrant** (Docker / cloud) | Skip re-index across runs | None | Iterative tuning |
| **INT8 scalar quantization** | Smaller index, faster search | Minimal | 1M+ ([ADR-006](../decisions/0006-vector-storage-and-indexing.md)) |
| **Streaming ingest** (no full chunk list) | Enables 10M at all | None | Required before 10M |

### Where to run hyperscale (10M+)

| Option | Pros | Cons |
|--------|------|------|
| **Local Mac (M-series)** | Free, good for 10k real + 100k fast | 15 GB RSS at 10k real; not 10M |
| **AWS `r6i.4xlarge` / `r7g.8xlarge`** | 128 GB RAM, predictable | Cost per soak run |
| **AWS `g5.xlarge` + GPU ONNX** | Real embeddings at 100k+ | Setup + $/hr |
| **Self-hosted Linux + Docker Compose Qdrant** | Matches production path | You manage hardware |
| **GitHub Actions large runner** | CI-gated regression | Expensive; keep to sample/SciFact + 100k fast |

**Recommendation:** Treat the Mac as the **quality lab** (10k real). Use a **cloud VM with 64–128 GB RAM** and persistent Qdrant for **1M/10M `--fast` stress**, then add **streaming ingest** before claiming 10M end-to-end.

### Engineering gaps before 10M claim

1. **Streaming benchmark ingest** — do not build `list[Chunk]` for entire corpus in memory.
2. **Rerank regression on FiQA** — understand before publishing rerank numbers at scale.
3. **RSS profiling** — 15 GB at 10k suggests memory work before 100k real.
4. **Published stress reports** — one markdown per tier (`fiqa-100k-fast.md`, …) same as [fiqa-10k.md](fiqa-10k.md).

---

## Reports

| Report | Command | Status |
|--------|---------|--------|
| [fiqa-10k.md](fiqa-10k.md) | `recall benchmark --dataset beir:fiqa --scale 10k --batch-size 128` | ✅ Complete |
| `fiqa-100k-fast.md` | `... --scale 100k --fast` | Planned |
| `fiqa-1m-fast.md` | `... --scale 1m --fast` (cloud VM) | Planned |
| `fiqa-10m-fast.md` | `... --scale 10m --fast` (cloud VM + streaming) | Planned |

---

## References

- [ADR-013: Real-world benchmark datasets](../decisions/0013-real-world-benchmark-datasets.md)
- [ADR-012: Multi-scale stress benchmarking](../decisions/0012-synthetic-corpus-and-scale-benchmarking.md)
- [DR-001, DR-002](../engineering-ledger/decisions.md)
- [LL-001, LL-002](../engineering-ledger/lessons.md)
- [AP-002](../engineering-ledger/attack-plans.md) — FiQA 10k scale benchmark (done)
