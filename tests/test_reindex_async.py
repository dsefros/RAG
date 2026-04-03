from threading import Lock

from src.application.reindex_manager import ReindexAlreadyRunningError, ReindexManager


class DummySessionFactory:
    def __init__(self, repo):
        self.repo = repo

    def __call__(self):
        repo = self.repo

        class Ctx:
            def __enter__(self_inner):
                return repo

            def __exit__(self_inner, *args):
                return False

        return Ctx()


class DummyJob:
    def __init__(self, job_id=1, status="queued"):
        self.id = job_id
        self.status = status


class DummyRepo:
    def __init__(self):
        self.jobs = {1: DummyJob(1)}

    def get_running(self):
        return None

    def create(self, triggered_by, status="queued"):
        return DummyJob(2, status)

    def get(self, job_id):
        return self.jobs.get(job_id, DummyJob(job_id))

    def mark_running(self, job):
        job.status = "running"

    def mark_done(self, job, report, activated_collection_name):
        job.status = "succeeded"

    def mark_failed(self, job, error, report):
        job.status = "failed"


class DummyExecutor:
    def submit(self, fn, *args):
        self.fn = fn
        self.args = args


class DummyQdrant:
    pass


class DummySettings:
    pass


def test_reindex_trigger_is_non_blocking(monkeypatch):
    repo = DummyRepo()
    monkeypatch.setattr("src.application.reindex_manager.ReindexJobRepository", lambda session: session)

    mgr = ReindexManager(
        settings=DummySettings(),
        qdrant=DummyQdrant(),
        session_factory=DummySessionFactory(repo),
        lock=Lock(),
        executor=DummyExecutor(),
    )
    job_id = mgr.trigger(triggered_by=1)
    assert job_id == 2


def test_reindex_rejects_if_running(monkeypatch):
    repo = DummyRepo()
    repo.get_running = lambda: DummyJob(7, "running")
    monkeypatch.setattr("src.application.reindex_manager.ReindexJobRepository", lambda session: session)

    mgr = ReindexManager(
        settings=DummySettings(),
        qdrant=DummyQdrant(),
        session_factory=DummySessionFactory(repo),
        lock=Lock(),
        executor=DummyExecutor(),
    )

    import pytest

    with pytest.raises(ReindexAlreadyRunningError):
        mgr.trigger(triggered_by=1)
