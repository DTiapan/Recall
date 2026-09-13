"""Helpers for normalizing and filtering document source URIs."""

from __future__ import annotations

import re
from pathlib import Path

_TEMP_NAME = re.compile(r"^tmp[a-z0-9]+(?:\.[a-z0-9]+)?$", re.IGNORECASE)
_TEMP_PATH_MARKERS = ("/tmp/", "/var/folders/", "/private/var/folders/", "\\temp\\")


def is_junk_source_uri(uri: str | None) -> bool:
    """True for tempfile paths and other non-user-facing ingest artifacts."""
    if not uri:
        return True
    text = str(uri)
    lowered = text.lower()
    if any(marker in lowered for marker in _TEMP_PATH_MARKERS):
        return True
    name = Path(text).name
    if name.startswith("tmp") and (len(name) <= 4 or name[3:4] in {".", "_", "-"} or name[3].isalnum()):
        return True
    if _TEMP_NAME.match(name):
        return True
    return False


def display_basename(uri: str | None) -> str:
    """User-facing filename from a stored source URI."""
    if not uri:
        return "Unknown"
    return Path(str(uri)).name or str(uri)
