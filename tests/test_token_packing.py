from pathlib import Path
from threading import BoundedSemaphore

from src.application.query_service import QueryService
from src.config.settings import Settings


class DummyBackend:
    name = "test"

    def generate(self, **kwargs):
        return "ok"

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def max_context_tokens(self) -> int:
        return 30


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


def test_token_budget_truncates_context(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    model = tmp_path / "m.gguf"
    model.write_text("x")
    settings = Settings(
        document_root=str(docs),
        llm_backend="llama_cpp",
        llama_cpp={"model_path": str(model)},
        retrieval={"max_context_tokens": 20, "reserved_output_tokens": 5},
        access_policy={"folder_defaults": {"product": ["support"]}},
    )
    svc = QueryService(settings, DummyQdrant(), DummyBackend(), DummyAudit(), DummyEmbed(), DummyReranker(), BoundedSemaphore(value=1))
    ctx, selected, warnings = svc._pack_context(
        [
            {"source_path": "a", "chunk_index": 0, "text": "one two three four five six", "doc_id": "1"},
            {"source_path": "b", "chunk_index": 1, "text": "seven eight nine ten eleven twelve", "doc_id": "2"},
        ]
    )
    assert len(selected) == 1
    assert "context_truncated" in warnings
