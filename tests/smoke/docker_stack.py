"""Helpers for Docker Compose smoke-test lifecycle management."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


COMPOSE_PROJECT = "recall-smoke-test"
DEFAULT_SMOKE_PORT = 18080


def docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        subprocess.run(
            ["docker", "info"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return True
    except (subprocess.SubprocessError, OSError):
        return False


def compose_files(repo_root: Path) -> list[str]:
    return [
        str(repo_root / "docker-compose.yml"),
        str(repo_root / "docker-compose.smoke.yml"),
    ]


def start_stack(repo_root: Path, smoke_port: int = DEFAULT_SMOKE_PORT) -> None:
    env = {"SMOKE_API_PORT": str(smoke_port)}
    subprocess.run(
        [
            "docker",
            "compose",
            *_compose_file_args(compose_files(repo_root)),
            "-p",
            COMPOSE_PROJECT,
            "up",
            "-d",
            "--build",
            "--wait",
        ],
        cwd=repo_root,
        check=True,
        env=_merged_env(env),
        timeout=600,
    )


def stop_stack(repo_root: Path) -> None:
    subprocess.run(
        [
            "docker",
            "compose",
            *_compose_file_args(compose_files(repo_root)),
            "-p",
            COMPOSE_PROJECT,
            "down",
            "-v",
            "--remove-orphans",
        ],
        cwd=repo_root,
        check=True,
        timeout=180,
    )


def wait_for_health(base_url: str, timeout_seconds: float = 120.0) -> None:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None

    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/v1/health", timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
                if response.status == 200 and payload.get("status") == "healthy":
                    return
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
        time.sleep(2)

    raise TimeoutError(f"Service at {base_url} did not become healthy: {last_error}")


def _compose_file_args(paths: list[str]) -> list[str]:
    args: list[str] = []
    for path in paths:
        args.extend(["-f", path])
    return args


def _merged_env(overrides: dict[str, str]) -> dict[str, str]:
    import os

    merged = os.environ.copy()
    merged.update(overrides)
    return merged
