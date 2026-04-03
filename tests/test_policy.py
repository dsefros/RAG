from pathlib import Path

from src.config.settings import Settings
from src.ingestion.policy import resolve_document_policy


def make_settings(root: Path, model: Path):
    return Settings(
        document_root=str(root),
        llm_backend="llama_cpp",
        llama_cpp={"model_path": str(model)},
        access_policy={"folder_defaults": {"product": ["support"], "regulatory": ["compliance"]}},
    )


def test_sidecar_overrides_defaults(tmp_path: Path):
    root = tmp_path / "docs"
    p = root / "product"
    p.mkdir(parents=True)
    f = p / "a.pdf"
    f.write_text("x")
    (p / "a.meta.yaml").write_text("allowed_groups: [ops]\nstatus: active\n")
    model = tmp_path / "m.gguf"
    model.write_text("x")

    settings = make_settings(root, model)
    policy = resolve_document_policy(settings, f)
    assert policy["allowed_groups"] == ["ops"]
    assert policy["status"] == "active"
