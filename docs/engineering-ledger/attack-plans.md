# Attack plans

## Active index

| ID | Title | Status |
|----|-------|--------|
| AP-002 | FiQA 10k scale benchmark (batched) | planned |

---

### AP-001 — Real-world eval + sample corpus (backfill)

- **Date:** 2026-09-01 – 2026-09-13
- **Status:** done
- **Goal:** Prove retrieval quality on real formats and BEIR SciFact
- **Problem framing:** Synthetic evals insufficient; need BEIR + PDF/DOCX sample corpus with rerank metrics
- **Components to attack:**
  1. Sample corpus (MD/PDF/DOCX) + queries
  2. BEIR SciFact loader with doc_id preservation
  3. Rerank metrics + config.yaml local defaults
  4. CI gate + README benchmark table
  5. Batched ingest for scale (batch_size=128, --fast mode)
- **Done when:** Tests green; README table; SciFact HitRate@5 > 80%
- **Links:** commits 096a041, 5e0f296, 96ed9f1, 2ea19cb; ADR-013

### AP-002 — FiQA 10k scale benchmark (batched)

- **Date:** 2026-09-13
- **Status:** planned
- **Goal:** Document throughput/RSS/P99 at 10k docs with batched ingest
- **Components to attack:**
  1. `recall benchmark --dataset beir:fiqa --scale 10k --batch-size 128`
  2. Update README benchmark table
  3. Record evidence in phases.md
- **Done when:** Benchmark completes; README + ledger updated
- **Links:** DR-002

---

## Archive

<!-- Move cancelled/superseded AP summaries here -->
