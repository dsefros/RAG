from __future__ import annotations

from sqlalchemy import inspect, text

from src.config.settings import Settings
from src.generation.backends import LLMBackend
from src.infrastructure.qdrant_store import QdrantStore
from src.persistence.db import build_engine


REQUIRED_TABLES = {
    "users",
    "groups",
    "user_group_memberships",
    "reindex_jobs",
    "audit_events",
}


class StartupValidationError(RuntimeError):
    pass


def validate_startup(settings: Settings, *, qdrant: QdrantStore, backend: LLMBackend) -> None:
    errors: list[str] = []

    try:
        engine = build_engine(settings)
        with engine.connect() as conn:
            conn.execute(text("select 1"))
        existing = set(inspect(engine).get_table_names())
        missing = sorted(REQUIRED_TABLES - existing)
        if missing:
            errors.append(f"Missing DB tables {missing}. Run: python scripts/run_migrations.py")
    except Exception as exc:
        errors.append(f"Postgres unavailable: {exc}")

    try:
        qdrant.ping()
        if qdrant.get_alias_target(settings.qdrant.active_alias) is None:
            errors.append(f"Qdrant alias '{settings.qdrant.active_alias}' is missing")
    except Exception as exc:
        errors.append(f"Qdrant unavailable: {exc}")

    try:
        ok, reason = backend.ready()
        if not ok:
            errors.append(f"Active backend not ready: {reason}")
    except Exception as exc:
        errors.append(f"Backend init failed: {exc}")

    if errors:
        raise StartupValidationError("; ".join(errors))
