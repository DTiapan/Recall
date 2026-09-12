"""Markdown document loader with heading hierarchy and frontmatter extraction."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from recall.core.interfaces import BaseDocumentLoader
from recall.core.models import Document
from recall.preprocessing.cleaner import clean_text

HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


class MarkdownLoader(BaseDocumentLoader):
    """Loads Markdown files, parsing YAML frontmatter and structural heading hierarchies."""

    def __init__(self, parse_frontmatter: bool = True) -> None:
        self.parse_frontmatter = parse_frontmatter

    def _extract_frontmatter(self, text: str) -> tuple[dict[str, Any], str]:
        metadata: dict[str, Any] = {}
        if not self.parse_frontmatter:
            return metadata, text

        match = FRONTMATTER_PATTERN.match(text)
        if match:
            raw_fm = match.group(1)
            content_body = text[match.end() :]
            for line in raw_fm.split("\n"):
                if ":" in line:
                    key, val = line.split(":", 1)
                    metadata[key.strip()] = val.strip().strip("\"'")
            return metadata, content_body

        return metadata, text

    def _extract_headings(self, text: str) -> list[dict[str, Any]]:
        headings = []
        for match in HEADING_PATTERN.finditer(text):
            level = len(match.group(1))
            title = match.group(2).strip()
            headings.append({"level": level, "title": title, "pos": match.start()})
        return headings

    def load(self, source: str) -> list[Document]:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Source file or directory not found: {source}")

        files = [path] if path.is_file() else list(path.glob("**/*.md"))
        documents: list[Document] = []

        for file_path in files:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                raw_text = f.read()

            fm_meta, body = self._extract_frontmatter(raw_text)
            headings = self._extract_headings(body)
            cleaned = clean_text(body)

            if cleaned:
                meta = {
                    "filename": file_path.name,
                    "extension": file_path.suffix,
                    "size_bytes": file_path.stat().st_size,
                    "headings": [h["title"] for h in headings],
                    **fm_meta,
                }
                documents.append(
                    Document(
                        content=cleaned,
                        source_uri=str(file_path.resolve()),
                        metadata=meta,
                    )
                )

        return documents
