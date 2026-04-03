from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from sqlalchemy.orm import sessionmaker

from src.config.settings import Settings
from src.infrastructure.qdrant_store import QdrantStore
from src.persistence.repositories import ReindexJobRepository
from src.reindex.service import ReindexService


class ReindexAlreadyRunningError(RuntimeError):
    pass


class ReindexManager:
    def __init__(
        self,
        *,
        settings: Settings,
        qdrant: QdrantStore,
        session_factory: sessionmaker,
        lock: Lock,
        executor: ThreadPoolExecutor,
    ):
        self.settings = settings
        self.qdrant = qdrant
        self.session_factory = session_factory
        self.lock = lock
        self.executor = executor

    def trigger(self, *, triggered_by: int) -> int:
        with self.session_factory() as session:
            repo = ReindexJobRepository(session)
            active = repo.get_running()
            if active:
                raise ReindexAlreadyRunningError(f"Reindex job {active.id} already {active.status}")
            job = repo.create(triggered_by=triggered_by, status="queued")

        self.executor.submit(self._run_job, job.id)
        return job.id

    def _run_job(self, job_id: int) -> None:
        with self.lock:
            with self.session_factory() as session:
                repo = ReindexJobRepository(session)
                job = repo.get(job_id)
                if not job:
                    return
                repo.mark_running(job)

            service = ReindexService(self.settings, self.qdrant)
            try:
                _, report = service.run()
                with self.session_factory() as session:
                    repo = ReindexJobRepository(session)
                    job = repo.get(job_id)
                    if job:
                        repo.mark_done(job, report, report.get("activated_index_version"))
            except Exception as exc:
                with self.session_factory() as session:
                    repo = ReindexJobRepository(session)
                    job = repo.get(job_id)
                    if job:
                        repo.mark_failed(job, str(exc), {})
