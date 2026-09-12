"""Real-world benchmark dataset loaders."""

from recall.eval.datasets.beir import load_beir_benchmark
from recall.eval.datasets.local import load_local_benchmark
from recall.eval.datasets.models import BenchmarkCorpus, BenchmarkQuery

__all__ = [
    "BenchmarkCorpus",
    "BenchmarkQuery",
    "load_local_benchmark",
    "load_beir_benchmark",
]
