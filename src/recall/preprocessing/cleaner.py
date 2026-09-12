"""Text cleaning, sanitization, and normalization utilities for robust document ingestion."""

from __future__ import annotations

import re
import unicodedata

# Unsafe control characters and zero-width characters often used in invisible prompt injection
ZERO_WIDTH_CHARS = re.compile(r"[\u200B\u200C\u200D\u200E\u200F\uFEFF\u202A-\u202E]")
NULL_BYTES = re.compile(r"\x00")
EXCESSIVE_BLANK_LINES = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Normalizes raw input text for indexing and chunking.
    
    Operations:
    1. Unicode NFKC normalization (replaces composite glyphs, ligatures, fullwidth chars).
    2. Removal of null bytes (prevents C-string truncation in DB engines).
    3. Removal of invisible zero-width and bidi override characters (anti-adversarial sanitization).
    4. Normalization of Windows/Mac carriage returns to standard Unix line breaks.
    5. Stripping per-line leading/trailing spaces and collapsing internal consecutive spaces.
    6. Collapsing 3+ consecutive newlines into 2.
    """
    if not text:
        return ""

    # 1. Normalize unicode
    text = unicodedata.normalize("NFKC", text)

    # 2. Strip null bytes
    text = NULL_BYTES.sub("", text)

    # 3. Strip zero-width & bidi characters
    text = ZERO_WIDTH_CHARS.sub("", text)

    # 4. Standardize newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 5. Clean each line: collapse internal spaces, strip leading/trailing whitespace
    cleaned_lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    text = "\n".join(cleaned_lines)

    # 6. Collapse excessive blank lines
    text = EXCESSIVE_BLANK_LINES.sub("\n\n", text)

    return text.strip()
