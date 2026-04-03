import pytest

from src.application.startup import StartupValidationError, validate_startup


class DummyQ:
    def ping(self):
        return None

    def get_alias_target(self, alias):
        return "x"


class DummyB:
    def ready(self):
        return True, "ok"


class DummySettings:
    qdrant = type("Q", (), {"active_alias": "rag_docs_active"})


def test_startup_fails_when_migrations_missing(monkeypatch):
    class DummyEngine:
        def connect(self):
            class Ctx:
                def __enter__(self_inner):
                    return self_inner

                def __exit__(self_inner, *args):
                    return False

                def execute(self_inner, *_args, **_kwargs):
                    return None

            return Ctx()

    monkeypatch.setattr("src.application.startup.build_engine", lambda _s: DummyEngine())
    monkeypatch.setattr("src.application.startup.inspect", lambda _e: type("I", (), {"get_table_names": lambda self: []})())

    with pytest.raises(StartupValidationError):
        validate_startup(DummySettings(), qdrant=DummyQ(), backend=DummyB())
