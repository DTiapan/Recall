"""Load real-world BEIR benchmark corpora with human relevance judgments."""

from __future__ import annotations

from pathlib import Path

from recall.eval.datasets.models import BenchmarkCorpus, BenchmarkDocument, BenchmarkQuery

SUPPORTED_BEIR_DATASETS = {
    "scifact",
    "nfcorpus",
    "fiqa",
    "arguana",
    "quora",
    "scidocs",
    "trec-covid",
}


def load_beir_benchmark(
    dataset_name: str,
    limit: int | None = None,
    query_limit: int | None = 50,
    data_dir: Path | None = None,
) -> BenchmarkCorpus:
    """Downloads (if needed) and loads a BEIR dataset with real documents and qrels."""
    normalized = dataset_name.lower().strip()
    if normalized not in SUPPORTED_BEIR_DATASETS:
        supported = ", ".join(sorted(SUPPORTED_BEIR_DATASETS))
        raise ValueError(f"Unsupported BEIR dataset '{dataset_name}'. Supported: {supported}")

    try:
        from beir import util
        from beir.datasets.data_loader import GenericDataLoader
    except ImportError as exc:
        raise ImportError(
            "BEIR datasets require the benchmark extra: uv pip install -e '.[benchmark]'"
        ) from exc

    cache_root = data_dir or Path.home() / ".cache" / "recall" / "beir"
    cache_root.mkdir(parents=True, exist_ok=True)
    dataset_path = util.download_and_unzip(
        f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{normalized}.zip",
        str(cache_root),
    )

    corpus, queries, qrels = GenericDataLoader(data_folder=dataset_path).load(split="test")

    documents: list[BenchmarkDocument] = []
    for doc_id, record in corpus.items():
        title = record.get("title", "")
        text = record.get("text", "")
        body = f"{title}\n\n{text}".strip() if title else text
        documents.append(
            BenchmarkDocument(
                doc_id=str(doc_id),
                text=body,
                source_uri=f"beir:{normalized}/{doc_id}",
                metadata={"dataset": normalized},
            )
        )
        if limit is not None and len(documents) >= limit:
            break

    allowed_doc_ids = {doc.doc_id for doc in documents}
    benchmark_queries: list[BenchmarkQuery] = []
    for query_id, query_text in queries.items():
        relevant = [doc_id for doc_id, score in qrels.get(query_id, {}).items() if score > 0]
        relevant = [doc_id for doc_id in relevant if doc_id in allowed_doc_ids]
        if not relevant:
            continue
        benchmark_queries.append(
            BenchmarkQuery(
                query_id=str(query_id),
                query=query_text,
                relevant_doc_ids=relevant,
            )
        )
        if query_limit is not None and len(benchmark_queries) >= query_limit:
            break

    return BenchmarkCorpus(
        name=f"beir:{normalized}",
        documents=documents,
        queries=benchmark_queries,
        data_root=Path(dataset_path),
    )
