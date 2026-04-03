from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ApiSettings(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    request_timeout_seconds: int = 60
    max_query_chars: int = 4000
    generation_concurrency: int = 4
    generation_acquire_timeout_seconds: int = 5


class PostgresSettings(BaseModel):
    dsn: str = "postgresql+psycopg://postgres:postgres@localhost:5432/rag"


class QdrantSettings(BaseModel):
    url: str = "http://localhost:6333"
    timeout_seconds: int = 30
    active_alias: str = "rag_docs_active"


class LlamaCppSettings(BaseModel):
    model_path: str = "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    n_ctx: int = 4096
    n_gpu_layers: int = -1
    n_threads: int = 8
    temperature: float = 0.1
    max_tokens: int = 512


class OllamaSettings(BaseModel):
    base_url: str = "http://localhost:11434"
    model: str = "mistral"
    temperature: float = 0.1
    max_tokens: int = 512
    context_window: int = 4096


class RetrievalSettings(BaseModel):
    embedding_model_name: str = "BAAI/bge-m3"
    reranker_model_name: str = "BAAI/bge-reranker-v2-m3"
    initial_top_k: int = 40
    rerank_top_k: int = 12
    final_top_k: int = 8
    max_context_tokens: int = 3000
    reserved_output_tokens: int = 512


class ChunkingSettings(BaseModel):
    chunk_size_tokens: int = 350
    chunk_overlap_tokens: int = 60
    min_chunk_chars: int = 30


class ReindexSettings(BaseModel):
    keep_last_versions: int = 2
    validate_min_points: int = 1
    upsert_batch_size: int = Field(default=64, ge=1)


class AuthSettings(BaseModel):
    jwt_secret: str = "change-me"
    jwt_algorithm: str = "HS256"
    token_ttl_minutes: int = 120


class LoggingSettings(BaseModel):
    level: str = "INFO"


class AccessPolicySettings(BaseModel):
    folder_defaults: Dict[str, List[str]] = Field(default_factory=dict)
    allowed_statuses: List[str] = Field(default_factory=lambda: ["active", "draft", "disabled"])

    @field_validator("folder_defaults")
    @classmethod
    def _validate_folder_defaults(cls, value: Dict[str, List[str]]) -> Dict[str, List[str]]:
        for folder, groups in value.items():
            if not folder.strip():
                raise ValueError("folder_defaults keys cannot be empty")
            if not groups:
                raise ValueError(f"folder '{folder}' must have at least one group")
        return value


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_nested_delimiter="__", extra="ignore")

    environment: Literal["dev", "test", "prod"] = "dev"
    document_root: str = "data/docs"
    llm_backend: Literal["llama_cpp", "ollama"] = "llama_cpp"

    api: ApiSettings = Field(default_factory=ApiSettings)
    postgres: PostgresSettings = Field(default_factory=PostgresSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    llama_cpp: LlamaCppSettings = Field(default_factory=LlamaCppSettings)
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    reindex: ReindexSettings = Field(default_factory=ReindexSettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    access_policy: AccessPolicySettings = Field(default_factory=AccessPolicySettings)

    @model_validator(mode="after")
    def _validate_runtime(self) -> "Settings":
        if not Path(self.document_root).exists():
            raise ValueError(f"document_root does not exist: {self.document_root}")

        if self.llm_backend == "llama_cpp" and not Path(self.llama_cpp.model_path).exists():
            raise ValueError(f"llama_cpp.model_path does not exist: {self.llama_cpp.model_path}")

        if self.llm_backend == "ollama" and not self.ollama.model:
            raise ValueError("ollama.model must be set")

        if not self.access_policy.folder_defaults:
            raise ValueError("access_policy.folder_defaults must not be empty")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
