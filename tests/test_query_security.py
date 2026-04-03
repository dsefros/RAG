from pathlib import Path
from threading import BoundedSemaphore

from src.application.query_service import QueryService
from src.config.settings import Settings


class DummyQdrant:
    def search(self, **kwargs):
        class Hit:
            def __init__(self, payload, score):
                self.payload = payload
                self.score = score

        return [
            Hit({"text": "allowed", "doc_id": "1", "source_path": "a", "chunk_index": 0, "allowed_groups": ["support"], "status": "active", "index_version": "v1"}, 0.9),
            Hit({"text": "forbidden", "doc_id": "2", "source_path": "b", "chunk_index": 1, "allowed_groups": ["finance"], "status": "active", "index_version": "v1"}, 0.95),
        ]


class DummyBackend:
    name = "test"

    def generate(self, **kwargs):
        assert "forbidden" not in kwargs["prompt"]
        return "ok"

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def max_context_tokens(self) -> int:
        return 256


class DummyAudit:
    def log(self, **kwargs):
        return None


class DummyEmbed:
    def encode(self, *args, **kwargs):
        return [[0.1, 0.2, 0.3]]


class DummyReranker:
    def rerank(self, query, docs, top_k):
        return docs


def test_forbidden_chunk_never_reaches_prompt(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    model = tmp_path / "m.gguf"
    model.write_text("x")
    settings = Settings(
        document_root=str(docs),
        llm_backend="llama_cpp",
        llama_cpp={"model_path": str(model)},
        access_policy={"folder_defaults": {"product": ["support"]}},
    )

    svc = QueryService(
        settings,
        DummyQdrant(),
        DummyBackend(),
        DummyAudit(),
        DummyEmbed(),
        DummyReranker(),
        BoundedSemaphore(value=1),
    )
    result = svc.answer(user_id=1, groups=["support"], question="q")
    assert result.answer == "ok"
    assert all(src["doc_id"] != "2" for src in result.sources)
