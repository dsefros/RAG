from pathlib import Path

import pytest

from src.config.settings import Settings


def test_config_validation_requires_folder_defaults(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    model = tmp_path / "m.gguf"
    model.write_text("x")
    with pytest.raises(ValueError):
        Settings(document_root=str(docs), llm_backend="llama_cpp", llama_cpp={"model_path": str(model)}, access_policy={"folder_defaults": {}})
