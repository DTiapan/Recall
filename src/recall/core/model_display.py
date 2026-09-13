"""Human-readable labels for LLM model identifiers."""

from __future__ import annotations


def model_display_name(model_id: str | None) -> str:
    """Convert an OpenRouter or LiteLLM model slug to a UI label."""
    if not model_id:
        return "Unknown"
    raw = str(model_id)
    if raw.startswith("openrouter/"):
        raw = raw.removeprefix("openrouter/")
    slug = raw.split("/")[-1]
    if slug.startswith("gpt-oss"):
        return slug.upper().replace("-", " ")
    return slug.replace("-", " ").title()
