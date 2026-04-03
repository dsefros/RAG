from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.api.dependencies import get_current_user, get_db, get_effective_groups, get_runtime_container, get_session_factory, require_admin
from src.api.schemas import (
    GroupCreateRequest,
    LoginRequest,
    LoginResponse,
    MembershipRequest,
    QueryRequest,
    QueryResponse,
    ReindexResponse,
    UserCreateRequest,
)
from src.application.query_service import OverloadedError, QueryService
from src.application.reindex_manager import ReindexAlreadyRunningError, ReindexManager
from src.application.runtime import RuntimeContainer, build_runtime_container
from src.application.startup import validate_startup
from src.auth.security import create_access_token, hash_password, verify_password
from src.config.settings import get_settings
from src.persistence.db import build_session_factory
from src.persistence.repositories import AuditRepository, GroupRepository, MembershipRepository, ReindexJobRepository, UserRepository


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    session_factory = build_session_factory(settings)
    container = build_runtime_container(settings)

    validate_startup(settings, qdrant=container.qdrant, backend=container.backend)

    app.state.container = container
    app.state.session_factory = session_factory
    yield
    container.reindex_executor.shutdown(wait=False)


app = FastAPI(title="Sommers RAG API", version="1.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready(container: RuntimeContainer = Depends(get_runtime_container), db: Session = Depends(get_db)):
    failures = []
    try:
        db.execute(text("select 1"))
    except Exception as exc:
        failures.append(f"postgres: {exc}")

    try:
        container.qdrant.ping()
        if container.qdrant.get_alias_target(container.settings.qdrant.active_alias) is None:
            failures.append("qdrant_alias_missing")
    except Exception as exc:
        failures.append(f"qdrant: {exc}")

    ok, reason = container.backend.ready()
    if not ok:
        failures.append(f"llm_backend: {reason}")

    if failures:
        raise HTTPException(status_code=503, detail=failures)
    return {"status": "ready"}


@app.post("/api/v1/auth/login", response_model=LoginResponse)
def login(payload: LoginRequest, container: RuntimeContainer = Depends(get_runtime_container), db: Session = Depends(get_db)):
    user = UserRepository(db).by_username(payload.username)
    if not user or not user.is_active or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token(container.settings, user_id=user.id, username=user.username, is_admin=user.is_admin)
    return LoginResponse(access_token=token)


@app.post("/api/v1/query", response_model=QueryResponse)
def query(
    payload: QueryRequest,
    user=Depends(get_current_user),
    groups: list[str] = Depends(get_effective_groups),
    container: RuntimeContainer = Depends(get_runtime_container),
    db: Session = Depends(get_db),
):
    if len(payload.question) > container.settings.api.max_query_chars:
        raise HTTPException(status_code=400, detail="Question too long")

    svc = QueryService(
        settings=container.settings,
        qdrant=container.qdrant,
        backend=container.backend,
        audit_repo=AuditRepository(db),
        embedder=container.embedder,
        reranker=container.reranker,
        generation_gate=container.generation_gate,
    )
    try:
        result = svc.answer(user_id=user.id, groups=groups, question=payload.question)
    except OverloadedError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    return QueryResponse(**result.__dict__)


@app.post("/api/v1/admin/reindex", response_model=ReindexResponse)
def admin_reindex(
    admin_user=Depends(require_admin),
    container: RuntimeContainer = Depends(get_runtime_container),
    session_factory=Depends(get_session_factory),
):
    manager = ReindexManager(
        settings=container.settings,
        qdrant=container.qdrant,
        session_factory=session_factory,
        lock=container.reindex_lock,
        executor=container.reindex_executor,
    )
    try:
        job_id = manager.trigger(triggered_by=admin_user.id)
    except ReindexAlreadyRunningError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return ReindexResponse(job_id=job_id, status="queued")


@app.get("/api/v1/admin/reindex/{job_id}")
def get_reindex(job_id: int, _: object = Depends(require_admin), db: Session = Depends(get_db)):
    job = ReindexJobRepository(db).get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return {
        "job_id": job.id,
        "status": job.status,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "report": job.report,
        "error_message": job.error_message,
    }


@app.post("/api/v1/admin/users")
def create_user(payload: UserCreateRequest, _: object = Depends(require_admin), db: Session = Depends(get_db)):
    user = UserRepository(db).create(payload.username, hash_password(payload.password), payload.is_admin, payload.is_active)
    return {"id": user.id, "username": user.username}


@app.get("/api/v1/admin/users")
def list_users(_: object = Depends(require_admin), db: Session = Depends(get_db)):
    users = UserRepository(db).list()
    return [{"id": u.id, "username": u.username, "is_admin": u.is_admin, "is_active": u.is_active} for u in users]


@app.post("/api/v1/admin/groups")
def create_group(payload: GroupCreateRequest, _: object = Depends(require_admin), db: Session = Depends(get_db)):
    g = GroupRepository(db).create(payload.name)
    return {"id": g.id, "name": g.name}


@app.get("/api/v1/admin/groups")
def list_groups(_: object = Depends(require_admin), db: Session = Depends(get_db)):
    return [{"id": g.id, "name": g.name} for g in GroupRepository(db).list()]


@app.post("/api/v1/admin/memberships")
def add_membership(payload: MembershipRequest, _: object = Depends(require_admin), db: Session = Depends(get_db)):
    users = UserRepository(db)
    groups = GroupRepository(db)
    user = users.by_username(payload.username)
    group = groups.by_name(payload.group_name)
    if not user or not group:
        raise HTTPException(status_code=404, detail="user/group not found")
    MembershipRepository(db).add(user.id, group.id)
    return {"status": "ok"}
