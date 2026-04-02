import pytest

from src.application.startup import StartupValidationError, validate_startup


class DummyQdrant:
    def ping(self):
        raise RuntimeError("qdrant")

    def get_alias_target(self, alias):
        return None


class DummyBackend:
    def ready(self):
        return False, "backend"


class DummySettings:
    qdrant = type("Q", (), {"active_alias": "rag_docs_active"})


def test_validate_startup_collects_errors(monkeypatch):
    monkeypatch.setattr("src.application.startup.build_engine", lambda s: (_ for _ in ()).throw(RuntimeError("db")))

    with pytest.raises(StartupValidationError):
        validate_startup(DummySettings(), qdrant=DummyQdrant(), backend=DummyBackend())
