"""Tests for unified configuration engine (.env + config.yaml)."""

from pathlib import Path
import pytest
from recall.core.config import PipelineConfig, load_config


def test_load_config_defaults(tmp_path: Path):
    # Non-existent config file should fall back safely to defaults
    cfg = load_config(config_path=tmp_path / "nonexistent.yaml")
    assert cfg.mode in ("local", "cloud")
    assert cfg.pipeline.ingest.chunk_size == 500
    assert cfg.pipeline.storage.collection_name == "enterprise_knowledge"
    assert cfg.pipeline.retrieval.rrf_k == 60


def test_load_config_from_yaml(tmp_path: Path):
    custom_yaml = tmp_path / "test_config.yaml"
    custom_yaml.write_text(
        """
version: "1.0"
mode: "local"
ingest:
  chunk_size: 250
  chunk_overlap: 25
retrieval:
  top_k: 10
  dense_weight: 0.7
  sparse_weight: 0.3
""",
        encoding="utf-8",
    )

    cfg = load_config(config_path=custom_yaml)
    assert cfg.mode == "local"
    assert cfg.pipeline.ingest.chunk_size == 250
    assert cfg.pipeline.ingest.chunk_overlap == 25
    assert cfg.pipeline.retrieval.top_k == 10
    assert cfg.pipeline.retrieval.dense_weight == 0.7
    # Active profile helpers
    assert cfg.active_embedding.provider == "fastembed"
    assert cfg.active_synthesis.provider == "ollama"


def test_env_overrides_yaml_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    custom_yaml = tmp_path / "config.yaml"
    custom_yaml.write_text("mode: local\n", encoding="utf-8")

    # Set env var override
    monkeypatch.setenv("RAG_MODE", "cloud")
    monkeypatch.setenv("API_PORT", "9000")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-mock-key")

    cfg = load_config(config_path=custom_yaml)
    # Env takes precedence
    assert cfg.mode == "cloud"
    assert cfg.env.api_port == 9000
    assert cfg.env.openai_api_key == "sk-test-mock-key"
    assert cfg.active_embedding.provider == "openai"
    assert cfg.active_synthesis.provider == "litellm"
