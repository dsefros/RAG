from src.infrastructure.qdrant_store import QdrantStore
import pytest


class DummyAlias:
    def __init__(self, alias_name: str, collection_name: str):
        self.alias_name = alias_name
        self.collection_name = collection_name


class DummyAliasResponse:
    def __init__(self, aliases):
        self.aliases = aliases


class DummyClient:
    def __init__(self):
        self.calls = []
        self.aliases = []
        self.upsert_calls = []
        self.fail_on_upsert_call = None

    def get_aliases(self):
        return DummyAliasResponse(self.aliases)

    def update_collection_aliases(self, change_aliases_operations):
        self.calls.append(change_aliases_operations)

    def upsert(self, collection_name, points):
        self.upsert_calls.append((collection_name, points))
        if self.fail_on_upsert_call == len(self.upsert_calls):
            raise RuntimeError("upsert failed")


def test_access_filter_uses_match_any_and_status():
    store = object.__new__(QdrantStore)
    filt = store.build_access_filter(["support", "ops"], status="active")
    assert filt.must[0].key == "allowed_groups"
    assert filt.must[1].key == "status"


def test_switch_alias_uses_create_and_delete_operations():
    store = object.__new__(QdrantStore)
    store.client = DummyClient()
    store.client.aliases = [DummyAlias("rag_docs_active", "rag_docs_v_old")]

    store.switch_alias("rag_docs_active", "rag_docs_v_new")

    ops = store.client.calls[-1]
    assert len(ops) == 2
    assert ops[0].__class__.__name__ == "DeleteAliasOperation"
    assert ops[1].__class__.__name__ == "CreateAliasOperation"


def test_ensure_alias_creates_when_missing():
    store = object.__new__(QdrantStore)
    store.client = DummyClient()

    store.ensure_alias("rag_docs_active", "rag_docs_v_init")

    ops = store.client.calls[-1]
    assert len(ops) == 1
    assert ops[0].__class__.__name__ == "CreateAliasOperation"


def test_upsert_batched_splits_into_expected_chunks():
    store = object.__new__(QdrantStore)
    store.client = DummyClient()
    points = list(range(250))

    completed = store.upsert_batched("rag_docs_v_new", points, batch_size=100)

    assert completed == 3
    assert [len(call[1]) for call in store.client.upsert_calls] == [100, 100, 50]


def test_upsert_batched_stops_processing_on_failure():
    store = object.__new__(QdrantStore)
    store.client = DummyClient()
    store.client.fail_on_upsert_call = 2
    points = list(range(250))

    with pytest.raises(RuntimeError):
        store.upsert_batched("rag_docs_v_new", points, batch_size=100)

    assert len(store.client.upsert_calls) == 2


def test_upsert_batched_rejects_invalid_batch_size():
    store = object.__new__(QdrantStore)
    store.client = DummyClient()

    with pytest.raises(ValueError):
        store.upsert_batched("rag_docs_v_new", [1, 2], batch_size=0)
