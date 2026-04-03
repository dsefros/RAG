from src.reindex.service import ReindexService


class DummyPipeline:
    def collect_documents(self):
        return ["a"]

    def build_points(self, files, version):
        return [1, 2], {"files_found": 1}


class DummyQdrant:
    def __init__(self):
        self.alias = "old"

    def create_collection(self, name, vector_size):
        self.created = name

    def upsert(self, name, points):
        self.upserted = len(points)

    def count(self, name):
        return 2

    def get_alias_target(self, alias):
        return self.alias

    def switch_alias(self, alias, new_collection):
        self.alias = new_collection


def test_reindex_switches_alias(tmp_path):
    from pathlib import Path
    from src.config.settings import Settings

    docs = tmp_path / "docs"
    docs.mkdir()
    model = tmp_path / "m.gguf"
    model.write_text("x")
    settings = Settings(document_root=str(docs), llm_backend="llama_cpp", llama_cpp={"model_path": str(model)}, access_policy={"folder_defaults": {"product": ["support"]}})
    q = DummyQdrant()
    svc = ReindexService(settings, q)
    svc.pipeline = DummyPipeline()
    version, report = svc.run()
    assert q.alias == version
    assert report["previous_active_collection"] == "old"
