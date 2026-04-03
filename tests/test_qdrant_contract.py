from src.infrastructure.qdrant_store import QdrantStore


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

    def get_aliases(self):
        return DummyAliasResponse(self.aliases)

    def update_collection_aliases(self, change_aliases_operations):
        self.calls.append(change_aliases_operations)


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
