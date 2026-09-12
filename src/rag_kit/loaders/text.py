"""Plain text document loader with robust encoding detection."""

from __future__ import annotations

import os
from pathlib import Path
from rag_kit.core.interfaces import BaseDocumentLoader
from rag_kit.core.models import Document
from rag_kit.preprocessing.cleaner import clean_text


class TextLoader(BaseDocumentLoader):
    """Loads plain text files with automatic encoding fallbacks (UTF-8, UTF-8-SIG, Latin-1)."""

    def __init__(self, autodetect_encoding: bool = True) -> None:
        self.autodetect_encoding = autodetect_encoding

    def _read_file_content(self, file_path: Path) -> str:
        encodings = ["utf-8", "utf-8-sig", "latin-1"] if self.autodetect_encoding else ["utf-8"]
        for enc in encodings:
            try:
                with open(file_path, "r", encoding=enc) as f:
                    return f.read()
            except (UnicodeDecodeError, LookupError):
                continue
        raise ValueError(f"Unable to decode text file '{file_path}' with supported encodings.")

    def load(self, source: str) -> list[Document]:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Source file or directory not found: {source}")

        files = [path] if path.is_file() else list(path.glob("**/*.txt"))
        documents: list[Document] = []

        for file_path in files:
            raw_text = self._read_file_content(file_path)
            cleaned = clean_text(raw_text)
            if cleaned:
                documents.append(
                    Document(
                        content=cleaned,
                        source_uri=str(file_path.resolve()),
                        metadata={
                            "filename": file_path.name,
                            "extension": file_path.suffix,
                            "size_bytes": file_path.stat().st_size,
                        },
                    )
                )

        return documents
