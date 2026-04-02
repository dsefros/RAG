from pathlib import Path
from threading import BoundedSemaphore

import pytest

from src.application.query_service import OverloadedError, QueryService
from src.config.settings import Settings


class DummyBackend:
    name = "test"

    def generate(self, **kwargs):
        return "ok"

    def count_tokens(self, text: str) -> int:
        return 1

    def max_context_tokens(self) -> int:
        return 128


class DummyEmbed:
    def encode(self, *args, **kwargs):
        return [[0.1, 0.2, 0.3]]


class DummyReranker:
    def rerank(self, query, docs, top_k):
        return docs


class DummyQdrant:
    def search(self, **kwargs):
        return []


class DummyAudit:
    def log(self, **kwargs):
        return None


def test_generation_gate_timeout_returns_overload(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    model = tmp_path / "m.gguf"
    model.write_text("x")
    settings = Settings(
        document_root=str(docs),
        llm_backend="llama_cpp",
        llama_cpp={"model_path": str(model)},
        api={"generation_acquire_timeout_seconds": 0},
        access_policy={"folder_defaults": {"product": ["support"]}},
    )
    gate = BoundedSemaphore(value=1)
    gate.acquire()
    svc = QueryService(settings, DummyQdrant(), DummyBackend(), DummyAudit(), DummyEmbed(), DummyReranker(), gate)

    with pytest.raises(OverloadedError):
        svc.answer(user_id=1, groups=["support"], question="q")
