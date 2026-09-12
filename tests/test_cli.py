"""Unit tests for Recall CLI commands."""

import subprocess
import sys
import pytest


def test_cli_help_flag():
    result = subprocess.run(
        [sys.executable, "-m", "recall.cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Recall: The Open-Source, Turnkey Enterprise RAG Platform" in result.stdout
    assert "serve" in result.stdout
    assert "ingest" in result.stdout
    assert "query" in result.stdout


def test_cli_subcommand_help():
    for subcmd in ["serve", "ingest", "query"]:
        result = subprocess.run(
            [sys.executable, "-m", "recall.cli", subcmd, "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert f"usage: recall {subcmd}" in result.stdout
