"""Deduplication engines for exact match and fuzzy near-duplicate detection."""

from __future__ import annotations

import re
from typing import Iterable
import xxhash
import numpy as np

from recall.core.interfaces import BaseDeduplicator
from recall.core.models import Document, compute_content_hash
from recall.preprocessing.cleaner import clean_text


class ExactDeduplicator(BaseDeduplicator):
    """High-throughput exact deduplicator based on 64-bit xxhash content digests.
    
    Memory footprint: ~8 bytes per document, capable of holding 10M+ documents
    in < 100MB of RAM.
    """

    def __init__(self, initial_hashes: Iterable[str] | None = None) -> None:
        self._seen_hashes: set[str] = set(initial_hashes or [])

    @property
    def count(self) -> int:
        return len(self._seen_hashes)

    def is_duplicate(self, document: Document) -> bool:
        doc_hash = document.content_hash or compute_content_hash(document.content)
        return doc_hash in self._seen_hashes

    def register(self, document: Document) -> None:
        doc_hash = document.content_hash or compute_content_hash(document.content)
        self._seen_hashes.add(doc_hash)

    def filter_duplicates(self, documents: list[Document]) -> list[Document]:
        """Filters out duplicate documents and registers new unique ones."""
        unique_docs: list[Document] = []
        for doc in documents:
            if not self.is_duplicate(doc):
                self.register(doc)
                unique_docs.append(doc)
        return unique_docs


class NearDuplicateDetector:
    """Fast MinHash LSH near-duplicate detector based on character/word shingling.
    
    Identifies documents that have high Jaccard similarity (e.g. slight revisions,
    headers changed, or copy-pasted sections).
    """

    def __init__(
        self,
        num_perm: int = 64,
        threshold: float = 0.85,
        shingle_size: int = 3,
    ) -> None:
        self.num_perm = num_perm
        self.threshold = threshold
        self.shingle_size = shingle_size
        self._signatures: dict[str, np.ndarray] = {}

        # Precompute random seeds for hash permutations
        rng = np.random.RandomState(42)
        self._hash_seeds = rng.randint(0, 2**32 - 1, size=num_perm, dtype=np.uint32)

    def _get_shingles(self, text: str) -> set[str]:
        words = re.findall(r"\w+", clean_text(text).lower())
        if len(words) < self.shingle_size:
            return {" ".join(words)}
        return {
            " ".join(words[i : i + self.shingle_size])
            for i in range(len(words) - self.shingle_size + 1)
        }

    def compute_signature(self, text: str) -> np.ndarray:
        """Computes a MinHash signature vector for the input text."""
        shingles = self._get_shingles(text)
        if not shingles:
            return np.zeros(self.num_perm, dtype=np.uint32)

        # Hash each shingle into a uint32
        shingle_hashes = np.array(
            [xxhash.xxh32(s.encode("utf-8")).intdigest() for s in shingles],
            dtype=np.uint32,
        )

        # Compute min hash per permutation
        sig = np.empty(self.num_perm, dtype=np.uint32)
        for i in range(self.num_perm):
            seed = self._hash_seeds[i]
            permuted = shingle_hashes ^ seed
            sig[i] = np.min(permuted)

        return sig

    def estimate_similarity(self, sig1: np.ndarray, sig2: np.ndarray) -> float:
        """Estimates Jaccard similarity from two MinHash signatures."""
        if len(sig1) != len(sig2) or len(sig1) == 0:
            return 0.0
        return float(np.mean(sig1 == sig2))

    def find_near_duplicate(self, doc_id: str, text: str) -> tuple[str | None, float]:
        """Checks if text is near-duplicate of an already registered document.
        
        Returns: (duplicate_doc_id, estimated_similarity) or (None, 0.0)
        """
        sig = self.compute_signature(text)
        for existing_id, existing_sig in self._signatures.items():
            sim = self.estimate_similarity(sig, existing_sig)
            if sim >= self.threshold:
                return existing_id, sim

        self._signatures[doc_id] = sig
        return None, 0.0

    def is_duplicate(self, document: Document) -> bool:
        """Returns True if the document has a Jaccard similarity >= threshold with any registered document."""
        sig = self.compute_signature(document.content)
        for existing_sig in self._signatures.values():
            if self.estimate_similarity(sig, existing_sig) >= self.threshold:
                return True
        return False

    def register(self, document: Document) -> None:
        """Registers the document's MinHash signature."""
        sig = self.compute_signature(document.content)
        self._signatures[document.id] = sig
