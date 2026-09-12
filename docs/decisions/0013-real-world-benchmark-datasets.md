# ADR-013: Real-World Benchmark Datasets (BEIR + Curated Corpus)

## Status
Accepted

## Date
2026-09-13

## Context
ADR-012 introduced scale benchmarking with a streaming **synthetic** document generator. User and engineering feedback requires **real datasets** with authentic language, structure, and human relevance labels — not combinatorial templates.

Synthetic generation remains useful for stress-only soak tests, but it must not be the primary evaluation path.

## Decision

1. **Primary benchmark corpora are real-world datasets**
   - Bundled `data/sample/` markdown corpus with curated `eval/queries.jsonl`
   - Optional BEIR datasets (`scifact`, `fiqa`, `nfcorpus`, …) with human qrels

2. **Scale tiers subsample real corpora**
   - `--scale 10k|100k|1m|10m` applies document limits to downloaded BEIR collections
   - No synthetic text generation in the default benchmark path

3. **CLI**
   ```bash
   recall benchmark --dataset sample
   recall benchmark --dataset beir:scifact --limit 500
   recall benchmark --dataset beir:fiqa --scale 10k
   ```

4. **Metrics**
   - HitRate@5, MRR, ingest duration, query latency P50/P95

## Consequences
- Benchmark results are reproducible and credible for enterprise evaluators.
- BEIR requires `uv pip install -e ".[benchmark]"` and network on first download.
- ADR-012 synthetic generator is deferred; scale soak tests may revisit lightweight duplication of real docs later.

## Supersedes
Partially supersedes the synthetic generator emphasis in ADR-012 §2. Starter corpus and CLI goals remain valid.
