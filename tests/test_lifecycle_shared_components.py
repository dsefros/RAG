from src.application.runtime import RuntimeContainer


def test_runtime_container_reuses_singletons():
    # structural test for app-scoped singleton behavior
    gate = object()
    embed = object()
    reranker = object()
    backend = object()
    container = RuntimeContainer(
        settings=object(),
        qdrant=object(),
        backend=backend,
        embedder=embed,
        reranker=reranker,
        generation_gate=gate,
        reindex_lock=object(),
        reindex_executor=object(),
    )
    assert container.embedder is embed
    assert container.reranker is reranker
    assert container.backend is backend
    assert container.generation_gate is gate
