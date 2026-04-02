from __future__ import annotations

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


class SourceItem(BaseModel):
    doc_id: str
    source_path: str
    chunk_index: int
    score: float


class QueryResponse(BaseModel):
    request_id: str
    answer: str
    backend: str
    index_version: str | None
    duration_ms: int
    sources: list[SourceItem]
    warnings: list[str] = []


class ReindexResponse(BaseModel):
    job_id: int
    status: str


class UserCreateRequest(BaseModel):
    username: str
    password: str
    is_admin: bool = False
    is_active: bool = True


class GroupCreateRequest(BaseModel):
    name: str


class MembershipRequest(BaseModel):
    username: str
    group_name: str
