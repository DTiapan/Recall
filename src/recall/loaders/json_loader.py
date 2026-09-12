"""JSON and JSON-Lines document loader for enterprise data records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from recall.core.interfaces import BaseDocumentLoader
from recall.core.models import Document
from recall.preprocessing.cleaner import clean_text


class JSONLoader(BaseDocumentLoader):
    """Loads JSON arrays or JSONL files containing structured text and metadata."""

    def __init__(self, text_key: str = "text", id_key: str | None = "id") -> None:
        self.text_key = text_key
        self.id_key = id_key

    def load(self, source: str) -> list[Document]:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {source}")

        documents: list[Document] = []
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            if path.suffix == ".jsonl":
                records = [json.loads(line) for line in f if line.strip()]
            else:
                data = json.load(f)
                records = data if isinstance(data, list) else [data]

        for i, rec in enumerate(records):
            if not isinstance(rec, dict):
                continue
            text_val = rec.get(self.text_key, "")
            cleaned = clean_text(str(text_val))
            if not cleaned:
                continue

            doc_id = str(rec.get(self.id_key)) if self.id_key and self.id_key in rec else f"{path.stem}_{i}"
            meta = {k: v for k, v in rec.items() if k not in (self.text_key, self.id_key)}
            meta["source_uri"] = str(path.resolve())

            documents.append(
                Document(
                    id=doc_id,
                    content=cleaned,
                    source_uri=str(path.resolve()),
                    metadata=meta,
                )
            )

        return documents
