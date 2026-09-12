"""Smoke tests validating docker compose deployment of Recall."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

import pytest

from tests.smoke.docker_stack import (
    DEFAULT_SMOKE_PORT,
    docker_available,
    start_stack,
    stop_stack,
    wait_for_health,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


pytestmark = pytest.mark.smoke


@pytest.fixture(scope="module")
def smoke_base_url() -> str:
    if not docker_available():
        pytest.skip("Docker is not available in this environment")

    stop_stack(REPO_ROOT)
    start_stack(REPO_ROOT, smoke_port=DEFAULT_SMOKE_PORT)
    base_url = f"http://127.0.0.1:{DEFAULT_SMOKE_PORT}"
    try:
        wait_for_health(base_url)
        yield base_url
    finally:
        stop_stack(REPO_ROOT)


def _post_json(url: str, payload: dict) -> dict | list:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _post_form(url: str, form_data: dict[str, str]) -> dict:
    encoded = "&".join(f"{key}={urllib.parse.quote_plus(value)}" for key, value in form_data.items())
    request = urllib.request.Request(
        url,
        data=encoded.encode("utf-8"),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def test_smoke_health_endpoint(smoke_base_url: str):
    with urllib.request.urlopen(f"{smoke_base_url}/v1/health", timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))

    assert response.status == 200
    assert payload["status"] == "healthy"
    assert payload["service"] == "recall-kit"


def test_smoke_ingest_and_search_round_trip(smoke_base_url: str):
    ingest_payload = _post_form(
        f"{smoke_base_url}/v1/ingest",
        {
            "text": "Smoke test document about WireGuard VPN remote access requirements.",
            "source_uri": "smoke.md",
            "collection": "smoke_docs",
        },
    )
    assert ingest_payload["chunks_indexed"] >= 1

    search_results = _post_json(
        f"{smoke_base_url}/v1/search",
        {"query": "WireGuard VPN remote access", "limit": 5, "collection": "smoke_docs"},
    )
    assert isinstance(search_results, list)
    assert len(search_results) >= 1
    assert "WireGuard" in search_results[0]["text"]
