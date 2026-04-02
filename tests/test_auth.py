from pathlib import Path

from src.auth.security import create_access_token, decode_access_token, hash_password, verify_password
from src.config.settings import Settings


def test_password_and_jwt(tmp_path: Path):
    model = tmp_path / "m.gguf"
    model.write_text("x")
    docs = tmp_path / "docs"
    docs.mkdir()
    settings = Settings(
        document_root=str(docs),
        llm_backend="llama_cpp",
        llama_cpp={"model_path": str(model)},
        access_policy={"folder_defaults": {"product": ["support"]}},
    )
    pw = hash_password("secret")
    assert verify_password("secret", pw)
    token = create_access_token(settings, user_id=1, username="alice", is_admin=True)
    payload = decode_access_token(settings, token)
    assert payload["username"] == "alice"
