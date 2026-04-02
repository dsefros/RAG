from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import BoundedSemaphore, Lock

from sentence_transformers import SentenceTransformer

from src.config.settings import Settings
from src.generation.backends import LLMBackend, build_backend
from src.infrastructure.qdrant_store import QdrantStore
from src.persistence.db import build_engine
from src.retrieval.reranker import Reranker


@dataclass
class RuntimeContainer:
    settings: Settings
    qdrant: QdrantStore
    backend: LLMBackend
    embedder: SentenceTransformer
    reranker: Reranker
    generation_gate: BoundedSemaphore
    reindex_lock: Lock
    reindex_executor: ThreadPoolExecutor


def build_runtime_container(settings: Settings) -> RuntimeContainer:
    qdrant = QdrantStore(settings.qdrant.url, settings.qdrant.timeout_seconds)
    backend = build_backend(settings)
    embedder = SentenceTransformer(settings.retrieval.embedding_model_name, cache_folder="models", device="cuda")
    reranker = Reranker(settings.retrieval.reranker_model_name)
    gate = BoundedSemaphore(value=settings.api.generation_concurrency)

    return RuntimeContainer(
        settings=settings,
        qdrant=qdrant,
        backend=backend,
        embedder=embedder,
        reranker=reranker,
        generation_gate=gate,
        reindex_lock=Lock(),
        reindex_executor=ThreadPoolExecutor(max_workers=1, thread_name_prefix="reindex"),
    )
