# Benchmark Datasets

Recall benchmarks use **real-world documents and labeled queries**, not synthetic templates.

## Bundled sample corpus (`sample/`)

Three enterprise markdown documents (security policy, financial report, infrastructure runbook) plus `eval/queries.jsonl` with human-authored relevance labels.

```bash
recall benchmark --dataset sample
```

## BEIR datasets (optional)

Install the benchmark extra and run standard IR benchmarks:

```bash
uv pip install -e ".[benchmark]"
recall benchmark --dataset beir:scifact --limit 500
recall benchmark --dataset beir:fiqa --scale 10k
```

BEIR corpora download on first use (~tens of MB to GB depending on dataset) and are cached under `~/.cache/recall/beir/`.

## Custom local corpus

Point at any directory of `.md` / `.txt` files with optional `eval/queries.jsonl`:

```bash
recall benchmark --dataset-path /path/to/corpus
```
