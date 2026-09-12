# ADR-012: Synthetic Enterprise Corpus Generation and Multi-Scale Stress Benchmarking (10K to 10M Docs)

## Status
Accepted

## Date
2026-09-13

## Context
Prospective users and enterprise evaluators frequently lack ready-to-use, unencumbered document corpora for testing. Furthermore, enterprise validation demands proof that the architecture scales linearly across orders of magnitude:
- **10,000 documents** (SMB department / single workspace baseline)
- **100,000 documents** (Mid-market company knowledge base)
- **1,000,000 documents** (Large enterprise multi-division archive)
- **10,000,000 documents** (Hyperscale enterprise repository)

Challenges at 10M scale:
1. **Memory Ceiling**: Loading 1M to 10M documents in Python memory causes immediate Out-Of-Memory (OOM) crashes. Ingestion and benchmarking must stream from generator or disk iterators in bounded micro-batches (e.g. 500–1,000 chunks).
2. **Vector Space Consumption**: 10M $\times$ 384-dim float32 vectors require ~15.4 GB of raw vectors and >35 GB with uncompressed HNSW graphs. Scalar Quantization (INT8) is mandatory to fit in ~3.8 GB.
3. **Retrieval Fidelity under Scale**: As corpus size expands from 10K to 10M, candidate noise increases 1,000-fold. The benchmark must test "needle in a haystack" precision, MRR, and P95 latency across every tier.

## Decision

We implement:

```
                      Recall Scale Benchmark System
                                     │
         ┌───────────────────────────┴───────────────────────────┐
         ▼                                                       ▼
[Starter Document Corpus]                         [Streaming Synthetic Generator]
- data/sample/ (PDF, DOCX, MD)                    - IT, HR, Finance, Cloud, Legal
- Real tables, CVEs, policies                     - Seeded "needle" queries & answers
         │                                                       │
         └───────────────────────────┬───────────────────────────┘
                                     ▼
                   [Multi-Scale Ingestion Engine]
                   - Streaming Generator (zero OOM)
                   - Vector & Sparse Batched Pipelines
                   - Multi-Tier: 10K, 100K, 1M, 10M
                                     │
                                     ▼
                   [Comprehensive Metrics Engine]
                   - Ingestion Throughput (docs/sec, tok/sec)
                   - Peak RAM (RSS) & Disk Index Footprint
                   - Latency Percentiles (P50, P95, P99)
                   - Needle Retrieval Accuracy (HitRate@5, MRR)
```

### 1. Curated Starter Document Corpus (`data/sample/`)
We provide a realistic sample set covering key enterprise formats:
- `information_security_policy.md` (compliance, access keys, tables, CVEs)
- `employee_handbook.docx` (HR policies, benefits table, remote work)
- `cloud_infrastructure_spec.pdf` (Kubernetes, AWS VPC, network ports)
- `financial_quarterly_report.md` (Markdown balance sheets, revenue metrics)

### 2. High-Performance Streaming Generator
Generates realistic domain-specific enterprise documents dynamically using combinatorial templates (IT security, database configs, financial tables, compliance policies) with zero external network downloads. Automatically injects known target "needles" at random percentile depths (10%, 50%, 90%) to benchmark retrieval quality under scale.

### 3. Scalable Benchmark CLI
Command `recall benchmark --scale [10k|100k|1m|10m] --tier-fast`:
- Supports synthetic streaming generation directly into vector + sparse indexes.
- Runs needle queries measuring P50/P95/P99 latency and MRR.
- Generates markdown and console benchmark tables.

## Consequences
- Users can immediately test Recall with zero setup by dragging starter docs from `data/sample/`.
- Engineering teams can rigorously measure hardware sizing, RAM footprint, and latency from 10K to 10M documents.
