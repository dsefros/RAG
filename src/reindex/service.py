from __future__ import annotations

from datetime import datetime

from src.config.settings import Settings
from src.ingestion.pipeline import IngestionPipeline
from src.infrastructure.qdrant_store import QdrantStore


class ReindexError(Exception):
    pass


class ReindexService:
    def __init__(self, settings: Settings, qdrant: QdrantStore):
        self.settings = settings
        self.qdrant = qdrant
        self.pipeline = IngestionPipeline(settings)

    def next_collection_name(self) -> str:
        return f"rag_docs_v{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    def run(self) -> tuple[str, dict]:
        version = self.next_collection_name()
        files = self.pipeline.collect_documents()
        points, report = self.pipeline.build_points(files, version)

        self.qdrant.create_collection(version, vector_size=1024)
        if points:
            self.qdrant.upsert(version, points)

        count = self.qdrant.count(version)
        report["points_in_collection"] = count
        report["new_collection"] = version

        if count < self.settings.reindex.validate_min_points:
            raise ReindexError(f"Validation failed: points={count}")

        old_active = self.qdrant.get_alias_target(self.settings.qdrant.active_alias)
        self.qdrant.switch_alias(self.settings.qdrant.active_alias, version)

        report["activated_index_version"] = version
        report["previous_active_collection"] = old_active
        return version, report

    def rollback(self, previous_collection: str) -> None:
        self.qdrant.switch_alias(self.settings.qdrant.active_alias, previous_collection)
