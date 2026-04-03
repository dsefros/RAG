from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from threading import BoundedSemaphore

from src.config.settings import Settings
from src.generation.backends import LLMBackend
from src.infrastructure.qdrant_store import QdrantStore
from src.persistence.repositories import AuditRepository
from src.retrieval.reranker import Reranker


class OverloadedError(RuntimeError):
    pass


@dataclass
class QueryResult:
    request_id: str
    answer: str
    backend: str
    index_version: str | None
    duration_ms: int
    sources: list[dict]
    warnings: list[str]


class QueryService:
    def __init__(
        self,
        settings: Settings,
        qdrant: QdrantStore,
        backend: LLMBackend,
        audit_repo: AuditRepository,
        embedder,
        reranker: Reranker,
        generation_gate: BoundedSemaphore,
    ):
        self.settings = settings
        self.qdrant = qdrant
        self.backend = backend
        self.audit_repo = audit_repo
        self.embedder = embedder
        self.reranker = reranker
        self.generation_gate = generation_gate

    def _pack_context(self, docs: list[dict]) -> tuple[str, list[dict], list[str]]:
        warnings = []
        selected = []
        lines: list[str] = []

        reserve_output = self.settings.retrieval.reserved_output_tokens
        max_ctx = min(self.settings.retrieval.max_context_tokens, self.backend.max_context_tokens() - reserve_output)
        used_tokens = self.backend.count_tokens("Системные инструкции и формат ответа.")

        for d in docs:
            fragment = f"[source={d['source_path']}#chunk={d['chunk_index']}]\n{d['text']}\n"
            fragment_tokens = self.backend.count_tokens(fragment)
            if used_tokens + fragment_tokens > max_ctx:
                warnings.append("context_truncated")
                break
            lines.append(fragment)
            selected.append(d)
            used_tokens += fragment_tokens

        return "\n".join(lines), selected, warnings

    def answer(self, *, user_id: int, groups: list[str], question: str) -> QueryResult:
        start = time.perf_counter()
        request_id = uuid.uuid4().hex
        warnings: list[str] = []

        vector = self.embedder.encode([question], convert_to_numpy=True, normalize_embeddings=True)[0].tolist()
        hits = self.qdrant.search(
            collection_alias=self.settings.qdrant.active_alias,
            vector=vector,
            top_k=self.settings.retrieval.initial_top_k,
            allowed_groups=groups,
        )
        candidates = []
        for h in hits:
            p = h.payload
            if not set(groups).intersection(set(p.get("allowed_groups", []))):
                continue
            if p.get("status") != "active":
                continue
            candidates.append({**p, "score": h.score})

        reranked = self.reranker.rerank(question, candidates, self.settings.retrieval.rerank_top_k)
        selected = reranked[: self.settings.retrieval.final_top_k]
        context, selected_for_prompt, pack_warnings = self._pack_context(selected)
        warnings.extend(pack_warnings)

        prompt = (
            "Ты — эксперт поддержки Sommers. Отвечай только на основе контекста.\n"
            "Если данных нет, скажи что в материалах нет информации.\n\n"
            f"Контекст:\n{context}\n\nВопрос: {question}\nОтвет:"
        )

        acquired = self.generation_gate.acquire(timeout=self.settings.api.generation_acquire_timeout_seconds)
        if not acquired:
            raise OverloadedError("Generation queue is full. Try again later.")

        try:
            answer = self.backend.generate(
                prompt=prompt,
                max_tokens=self.settings.retrieval.reserved_output_tokens,
                temperature=(self.settings.llama_cpp.temperature if self.settings.llm_backend == "llama_cpp" else self.settings.ollama.temperature),
            )
        finally:
            self.generation_gate.release()

        duration_ms = int((time.perf_counter() - start) * 1000)
        index_version = selected_for_prompt[0]["index_version"] if selected_for_prompt else None

        trace = {
            "groups": groups,
            "retrieved_chunks": [{"doc_id": d["doc_id"], "source_path": d["source_path"], "score": d["score"]} for d in candidates],
            "reranked_chunks": [{"doc_id": d["doc_id"], "source_path": d["source_path"], "rerank_score": d.get("rerank_score", 0)} for d in reranked],
            "selected_chunks": [{"doc_id": d["doc_id"], "source_path": d["source_path"], "chunk_index": d["chunk_index"]} for d in selected_for_prompt],
            "backend": self.backend.name,
            "index_version": index_version,
            "duration_ms": duration_ms,
            "warnings": warnings,
        }
        self.audit_repo.log(
            request_id=request_id,
            user_id=user_id,
            event_type="query",
            question_raw=question,
            answer_raw=answer,
            trace=trace,
        )

        sources = [
            {"doc_id": d["doc_id"], "source_path": d["source_path"], "chunk_index": d["chunk_index"], "score": d.get("rerank_score", d.get("score", 0.0))}
            for d in selected_for_prompt
        ]
        return QueryResult(
            request_id=request_id,
            answer=answer,
            backend=self.backend.name,
            index_version=index_version,
            duration_ms=duration_ms,
            sources=sources,
            warnings=warnings,
        )
