from pathlib import Path

from src.config.settings import Settings
from src.generation.backends import build_backend


class DummyTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [1, 2]


def test_backend_selection_ollama(tmp_path: Path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    monkeypatch.setattr("src.generation.backends.AutoTokenizer.from_pretrained", lambda *_a, **_k: DummyTokenizer())
    settings = Settings(
        document_root=str(docs),
        llm_backend="ollama",
        access_policy={"folder_defaults": {"product": ["support"]}},
    )
    backend = build_backend(settings)
    assert backend.name == "ollama"
