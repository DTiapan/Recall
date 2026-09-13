"""Unified Configuration Engine merging .env secrets and config.yaml pipeline parameters."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal
import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class IngestSettings(BaseModel):
    chunk_size: int = Field(default=500, description="Tokens per chunk")
    chunk_overlap: int = Field(default=50, description="Token overlap")
    tokenizer: str = Field(default="cl100k_base", description="Tiktoken encoding name")
    enable_deduplication: bool = Field(default=True, description="Deduplication toggle")
    near_dup_threshold: float = Field(default=0.85, description="MinHash similarity threshold")
    preserve_tables: bool = Field(default=True, description="Preserve table markdown headers")


class HNSWSettings(BaseModel):
    m: int = 16
    ef_construct: int = 100


class QuantizationSettings(BaseModel):
    enabled: bool = True
    type: str = "scalar"


class StorageSettings(BaseModel):
    collection_name: str = "enterprise_knowledge"
    vector_size: int = 1536
    distance: str = "Cosine"
    hnsw: HNSWSettings = Field(default_factory=HNSWSettings)
    quantization: QuantizationSettings = Field(default_factory=QuantizationSettings)


class ModelProfile(BaseModel):
    provider: str
    model_name: str | None = None
    model: str | None = None
    dimensions: int | None = None


class EmbeddingSettings(BaseModel):
    local: ModelProfile = Field(
        default_factory=lambda: ModelProfile(
            provider="fastembed",
            model_name="BAAI/bge-small-en-v1.5",
            dimensions=384,
        )
    )
    cloud: ModelProfile = Field(
        default_factory=lambda: ModelProfile(
            provider="openai",
            model_name="text-embedding-3-small",
            dimensions=1536,
        )
    )


class RetrievalSettings(BaseModel):
    top_k: int = 20
    dense_weight: float = 0.6
    sparse_weight: float = 0.4
    rrf_k: int = 60
    score_threshold: float = 0.35
    timeout_ms: int = 150


class RerankingSettings(BaseModel):
    enabled: bool = True
    top_n: int = 5
    candidate_k: int = Field(
        default=50,
        description="Hybrid pool size fed to the cross-encoder before reranking",
    )
    local_model: str = "ms-marco-MiniLM-L-12-v2"
    cloud_model: str = "cohere"
    timeout_ms: int = 250
    score_threshold: float | None = Field(
        default=None,
        description="Optional cross-encoder cutoff; None = rank only (recommended for eval)",
    )


class GuardrailsSettings(BaseModel):
    sandbox_xml_tags: bool = True
    enforce_citation: bool = True
    strict_refusal: bool = True


class ObservabilitySettings(BaseModel):
    enabled: bool = True
    service_name: str = "recall-kit"


class UIModelOption(BaseModel):
    id: str = Field(description="OpenRouter model slug, e.g. openai/gpt-oss-120b")
    label: str = Field(description="Human-readable label for the UI dropdown")


class UISettings(BaseModel):
    models: list[UIModelOption] = Field(default_factory=list)
    default_model: str | None = Field(
        default=None,
        description="Default UI model id; falls back to LOCAL_LLM_MODEL / server default",
    )


class SynthesisSettings(BaseModel):
    temperature: float = 0.1
    max_tokens: int = 1024
    local: ModelProfile = Field(
        default_factory=lambda: ModelProfile(
            provider="ollama",
            model="ollama/llama3:8b",
        )
    )
    cloud: ModelProfile = Field(
        default_factory=lambda: ModelProfile(
            provider="litellm",
            model="gpt-4o",
        )
    )
    guardrails: GuardrailsSettings = Field(default_factory=GuardrailsSettings)


class PipelineConfig(BaseModel):
    """Configuration loaded from config.yaml."""

    version: str = "1.0"
    mode: Literal["local", "cloud"] = "cloud"
    ingest: IngestSettings = Field(default_factory=IngestSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    reranking: RerankingSettings = Field(default_factory=RerankingSettings)
    synthesis: SynthesisSettings = Field(default_factory=SynthesisSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    ui: UISettings = Field(default_factory=UISettings)


class EnvSettings(BaseSettings):
    """Environment variables and secrets loaded from .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    rag_mode: Literal["local", "cloud"] | None = Field(default=None, alias="RAG_MODE")
    rag_env: str = Field(default="production", alias="RAG_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    web_ui_port: int = Field(default=3000, alias="WEB_UI_PORT")
    rag_api_key: str | None = Field(default=None, alias="RAG_API_KEY")

    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_path: str | None = Field(default=None, alias="QDRANT_PATH")
    qdrant_api_key: str | None = Field(default=None, alias="QDRANT_API_KEY")

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    cohere_api_key: str | None = Field(default=None, alias="COHERE_API_KEY")
    openrouter_api_key: str | None = Field(default=None, alias="OPENROUTER_API_KEY")

    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    local_llm_model: str | None = Field(
        default=None,
        alias="LOCAL_LLM_MODEL",
        description="Override synthesis model (e.g. openai/gpt-oss-120b)",
    )

    otel_exporter_otlp_endpoint: str | None = Field(default=None, alias="OTEL_EXPORTER_OTLP_ENDPOINT")
    otel_service_name: str | None = Field(default=None, alias="OTEL_SERVICE_NAME")
    otel_sdk_disabled: bool = Field(default=False, alias="OTEL_SDK_DISABLED")


class AppConfig:
    """Unified application configuration combining .env and config.yaml."""

    def __init__(self, env: EnvSettings, pipeline: PipelineConfig) -> None:
        self.env = env
        self.pipeline = pipeline
        # .env overrides config.yaml mode if explicitly specified
        if env.rag_mode is not None:
            self.mode: Literal["local", "cloud"] = env.rag_mode
        else:
            self.mode = self.pipeline.mode

    @property
    def active_embedding(self) -> ModelProfile:
        return self.pipeline.embedding.local if self.mode == "local" else self.pipeline.embedding.cloud

    @property
    def active_synthesis(self) -> ModelProfile:
        return self.pipeline.synthesis.local if self.mode == "local" else self.pipeline.synthesis.cloud


def load_config(
    config_path: str | Path | None = None,
    env_file: str | Path | None = None,
) -> AppConfig:
    """Loads and merges configuration from .env and config.yaml.

    Resolution order:
    1. Default values in Pydantic models.
    2. Overrides from config.yaml (if file exists).
    3. Overrides from environment variables / .env.
    """
    env_kwargs: dict[str, Any] = {}
    if env_file:
        env_kwargs["_env_file"] = env_file

    env = EnvSettings(**env_kwargs)

    path = Path(config_path) if config_path else Path("config.yaml")
    pipeline_data: dict[str, Any] = {}

    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                pipeline_data = loaded

    pipeline = PipelineConfig(**pipeline_data)
    return AppConfig(env=env, pipeline=pipeline)
