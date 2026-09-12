"""End-to-End Ingestion Pipeline Integration Test.

Validates the full chain:
Raw File -> Adapter / Loader -> Cleaner/Sanitizer -> Deduplication Gate -> Chunking -> Enriched Metadata Chain.
"""

from pathlib import Path
from recall.adapters import ChunkingAdapterRegistry
from recall.core.config import load_config
from recall.core.models import Document, IngestConfig
from recall.preprocessing.cleaner import clean_text
from recall.preprocessing.dedup import ExactDeduplicator, NearDuplicateDetector


def test_full_e2e_ingestion_pipeline(tmp_path: Path):
    # 1. Load pipeline configuration
    config = load_config()
    ingest_cfg = IngestConfig(
        chunk_size=config.pipeline.ingest.chunk_size,
        chunk_overlap=config.pipeline.ingest.chunk_overlap,
        enable_deduplication=config.pipeline.ingest.enable_deduplication,
    )

    # 2. Setup deduplicators
    exact_dedup = ExactDeduplicator()
    near_dedup = NearDuplicateDetector(threshold=config.pipeline.ingest.near_dup_threshold)

    # 3. Create a realistic enterprise document with dirty characters, tables, and frontmatter
    doc_path = tmp_path / "financial_q4_report.md"
    raw_content = (
        "---\n"
        "title: Q4 2024 Financial Performance & Cloud Revenue\n"
        "department: Finance & Strategy\n"
        "classification: Internal Confidential\n"
        "---\n\n"
        "# Executive Summary\u200b\n\n"  # Has zero-width space
        "The company achieved record operational margins in Q4 2024.\x00\n\n"  # Has null byte
        "## Cloud Segment Highlights\n\n"
        "Cloud adoption grew by 42% year over year across all enterprise tiers.\n\n"
        "| Business Unit | Q4 Revenue | YoY Growth | Margin |\n"
        "|---|---|---|---|\n"
        "| Enterprise Cloud | $450M | +42% | 31% |\n"
        "| AI Solutions | $210M | +88% | 24% |\n"
        "| Developer Tools | $95M | +18% | 19% |\n\n"
        "## Risk Factors\n\n"
        "Supply chain constraints on specialized AI accelerators may impact H1 delivery schedules.\n"
    )
    doc_path.write_text(raw_content, encoding="utf-8")

    # 4. Step 1: Adapter Loading & Format-Aware Ingestion
    registry = ChunkingAdapterRegistry()
    raw_chunks = registry.process(doc_path, config=ingest_cfg)
    assert len(raw_chunks) >= 2, "Document should produce multiple structured chunks"

    # 5. Step 2: Preprocessing, Sanitization, and Deduplication Verification
    for chunk in raw_chunks:
        # Verify text is clean (no null bytes or zero-width spaces remain)
        cleaned = clean_text(chunk.text)
        assert "\x00" not in cleaned
        assert "\u200b" not in cleaned

        # Verify provenance and structural metadata
        assert chunk.metadata.doc_id is not None
        assert chunk.metadata.file_type == "markdown"
        assert chunk.metadata.policy_name == "Q4 2024 Financial Performance & Cloud Revenue"
        assert len(chunk.metadata.section_hierarchy) > 0

    # 6. Step 3: Register in Deduplication Gate
    parent_doc = Document(content=clean_text(raw_content), source_uri=str(doc_path))
    assert not exact_dedup.is_duplicate(parent_doc), "First ingestion should not be duplicate"
    exact_dedup.register(parent_doc)
    near_dedup.register(parent_doc)

    # 7. Step 4: Duplicate Ingestion Protection
    duplicate_doc = Document(content=clean_text(raw_content), source_uri=str(doc_path))
    assert exact_dedup.is_duplicate(duplicate_doc), "Identical document must be blocked by exact dedup"

    # 8. Step 5: Near-duplicate Protection (95% identical copy with minor typo)
    near_duplicate_content = clean_text(raw_content) + " Minor footnote update."
    near_doc = Document(content=near_duplicate_content)
    assert near_dedup.is_duplicate(near_doc), "Near-duplicate must be detected by MinHash LSH"

    # 9. Step 6: Verify Table Chunk Quality
    table_chunks = [c for c in raw_chunks if c.metadata.content_type == "table"]
    assert len(table_chunks) >= 1, "Financial table must be preserved as structured table chunk"
    assert "| Enterprise Cloud" in table_chunks[0].text
    assert "| Q4 Revenue" in table_chunks[0].text
