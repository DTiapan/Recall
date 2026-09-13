"""Retrieval benchmark regression floors for CI and local gates."""

from __future__ import annotations

from dataclasses import dataclass

from recall.eval.retrieval_benchmark import RetrievalBenchmarkReport


@dataclass(frozen=True)
class RetrievalGateFloors:
    """Minimum acceptable retrieval metrics (ratios in [0, 1])."""

    hit_rate_at_5: float | None = None
    mrr: float | None = None
    ndcg_at_10: float | None = None


# Baseline: sample corpus @ 100% HR@5, ~0.66 MRR (RAG_ENV=test mocks, Sep 2026).
SAMPLE_CORPUS_FLOORS = RetrievalGateFloors(hit_rate_at_5=0.90, mrr=0.60)


class RetrievalGateError(Exception):
    """Raised when a benchmark report falls below configured floors."""


def check_report_meets_floors(
    report: RetrievalBenchmarkReport,
    floors: RetrievalGateFloors,
    *,
    dataset_label: str | None = None,
) -> None:
    """Raise RetrievalGateError if any configured floor is not met."""
    label = dataset_label or report.dataset_name
    failures: list[str] = []

    if floors.hit_rate_at_5 is not None and report.hit_rate_at_5 < floors.hit_rate_at_5:
        failures.append(
            f"HitRate@5 {report.hit_rate_at_5:.1%} < floor {floors.hit_rate_at_5:.1%}"
        )
    if floors.mrr is not None and report.mrr < floors.mrr:
        failures.append(f"MRR {report.mrr:.3f} < floor {floors.mrr:.3f}")
    if floors.ndcg_at_10 is not None:
        if report.ndcg_at_10 is None:
            failures.append("nDCG@10 missing but floor was configured")
        elif report.ndcg_at_10 < floors.ndcg_at_10:
            failures.append(
                f"nDCG@10 {report.ndcg_at_10:.3f} < floor {floors.ndcg_at_10:.3f}"
            )

    if failures:
        detail = "; ".join(failures)
        raise RetrievalGateError(f"Retrieval gate failed for {label}: {detail}")
