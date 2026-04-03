from src.reindex.service import ReindexService
import pytest


class DummyPipeline:
    def collect_documents(self):
        return ["a"]

    def build_points(self, files, version):
        return [1, 2], {"files_found": 1}


class DummyQdrant:
    def __init__(self, fail_on_batch: int | None = None):
        self.alias = "old"
        self.fail_on_batch = fail_on_batch
        self.upsert_batch_calls = 0
        self.last_completed_batches = 0

    def create_collection(self, name, vector_size):
        self.created = name

    def upsert_batched(self, name, points, batch_size):
        self.upsert_batch_calls += 1
        total_batches = (len(points) + batch_size - 1) // batch_size
        if self.fail_on_batch is not None and self.fail_on_batch <= total_batches:
            raise RuntimeError("batch upload failed")
        self.last_completed_batches = total_batches
        self.upserted = len(points)
        return total_batches

    def count(self, name):
        return 2

    def get_alias_target(self, alias):
        return self.alias

    def switch_alias(self, alias, new_collection):
        self.alias = new_collection


def test_reindex_switches_alias(tmp_path):
    from src.config.settings import Settings

    docs = tmp_path / "docs"
    docs.mkdir()
    model = tmp_path / "m.gguf"
    model.write_text("x")
    settings = Settings(
        document_root=str(docs),
        llm_backend="llama_cpp",
        llama_cpp={"model_path": str(model)},
        access_policy={"folder_defaults": {"product": ["support"]}},
        reindex={"upsert_batch_size": 1},
    )
    q = DummyQdrant()
    svc = object.__new__(ReindexService)
    svc.settings = settings
    svc.qdrant = q
    svc.pipeline = DummyPipeline()
    version, report = svc.run()
    assert q.alias == version
    assert report["previous_active_collection"] == "old"
    assert report["upsert_batch_size"] == 1
    assert report["total_points"] == 2
    assert report["total_batches"] == 2
    assert report["completed_batches"] == 2


def test_reindex_does_not_switch_alias_when_upload_fails(tmp_path):
    from src.config.settings import Settings

    docs = tmp_path / "docs"
    docs.mkdir()
    model = tmp_path / "m.gguf"
    model.write_text("x")
    settings = Settings(
        document_root=str(docs),
        llm_backend="llama_cpp",
        llama_cpp={"model_path": str(model)},
        access_policy={"folder_defaults": {"product": ["support"]}},
        reindex={"upsert_batch_size": 1},
    )
    q = DummyQdrant(fail_on_batch=2)
    svc = object.__new__(ReindexService)
    svc.settings = settings
    svc.qdrant = q
    svc.pipeline = DummyPipeline()

    with pytest.raises(RuntimeError):
        svc.run()

    assert q.alias == "old"
