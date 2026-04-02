from src.infrastructure.qdrant_store import QdrantStore


def test_access_filter_uses_match_any_and_status():
    store = object.__new__(QdrantStore)
    filt = store.build_access_filter(["support", "ops"], status="active")
    assert filt.must[0].key == "allowed_groups"
    assert filt.must[1].key == "status"
