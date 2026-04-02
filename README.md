# Sommers Internal RAG API (Tier 1 hardened baseline)

API-first internal RAG service for Sommers support scenarios.

## What was hardened
- Heavy runtime components are app-scoped singletons (embedder, reranker, LLM backend, Qdrant client).
- Global generation concurrency gate is process-wide and enforces overload behavior.
- Reindex trigger is non-blocking: background managed job with persisted status.
- Context packing is token-budgeted (not char-based) with deterministic truncation.
- Qdrant alias/filter operations were hardened to typed operations and typed metadata filters.
- Startup no longer auto-creates schema with `create_all()`; migration state is validated.

## Architecture
- `src/api/`: HTTP endpoints and dependency wiring
- `src/application/runtime.py`: app-scoped runtime container for shared heavy services
- `src/application/query_service.py`: query orchestration, token budget packing, audit trace
- `src/application/reindex_manager.py`: asynchronous managed reindex scheduling
- `src/application/startup.py`: startup checks (db schema, qdrant alias, backend readiness)
- `src/infrastructure/qdrant_store.py`: typed Qdrant search/filter/alias lifecycle
- `src/persistence/`: Postgres models + repositories

## Environment
Copy `.env.example` to `.env`.

Important settings:
- `API__GENERATION_CONCURRENCY`
- `API__GENERATION_ACQUIRE_TIMEOUT_SECONDS`
- `RETRIEVAL__MAX_CONTEXT_TOKENS`
- `RETRIEVAL__RESERVED_OUTPUT_TOKENS`
- `LLM_BACKEND` (`llama_cpp` / `ollama`)
- `DOCUMENT_ROOT`
- `POSTGRES__DSN`
- `QDRANT__URL`, `QDRANT__ACTIVE_ALIAS`
- `ACCESS_POLICY__FOLDER_DEFAULTS`

## Schema lifecycle
Schema source of truth is SQL migrations (`migrations/001_init.sql`).

Run explicitly:
```bash
python scripts/run_migrations.py
```

App startup validates required tables and fails clearly if migrations were not run.

## Bootstrap first admin
```bash
python scripts/bootstrap_admin.py --username admin --password 'admin123'
```

## Run service
```bash
pip install -r requirements.txt
python main.py
```

## Reindex flow
- `POST /api/v1/admin/reindex` now returns quickly with `status=queued`.
- Background worker updates states: `queued -> running -> succeeded/failed`.
- Only one active/queued reindex is allowed at a time (second trigger returns conflict).
- `GET /api/v1/admin/reindex/{job_id}` returns persisted state/report.

## Token-budget context packing
Context assembly uses backend-aware token counting:
- backend max context window
- reserved output token budget
- deterministic first-fit selection
- `context_truncated` warning when limit is reached

## Local smoke path
Use helper script after service startup:
```bash
BASE_URL=http://127.0.0.1:8000 \
ADMIN_USER=admin \
ADMIN_PASS=admin123 \
./scripts/smoke_flow.sh
```

Script runs:
1. `/health`
2. login
3. trigger reindex
4. poll job status
5. authenticated query

## Current limitations
- Reindex execution is in-process (single worker thread), not an external worker system.
- Rollback endpoint is still not exposed (service logic can switch alias back).
- Integration tests with real Postgres/Qdrant are not included yet.
