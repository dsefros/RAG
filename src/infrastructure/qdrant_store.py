from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    CreateAlias,
    DeleteAlias,
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    UpdateCollectionAliasesOperation,
    VectorParams,
)


@dataclass
class SearchHit:
    id: str
    score: float
    payload: dict[str, Any]


class QdrantStore:
    def __init__(self, url: str, timeout: int = 30):
        self.client = QdrantClient(url=url, timeout=timeout)

    def ping(self) -> None:
        self.client.get_collections()

    def collection_exists(self, name: str) -> bool:
        return self.client.collection_exists(collection_name=name)

    def create_collection(self, name: str, vector_size: int) -> None:
        if not self.collection_exists(name):
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    def upsert(self, collection: str, points: list[PointStruct]) -> None:
        self.client.upsert(collection_name=collection, points=points)

    def build_access_filter(self, allowed_groups: list[str], status: str = "active") -> Filter:
        return Filter(
            must=[
                FieldCondition(key="allowed_groups", match=MatchAny(any=allowed_groups)),
                FieldCondition(key="status", match=MatchValue(value=status)),
            ]
        )

    def search(self, *, collection_alias: str, vector: list[float], top_k: int, allowed_groups: list[str], status: str = "active") -> list[SearchHit]:
        filt = self.build_access_filter(allowed_groups=allowed_groups, status=status)
        response = self.client.query_points(collection_name=collection_alias, query=vector, limit=top_k, query_filter=filt)
        return [SearchHit(id=str(p.id), score=float(p.score), payload=p.payload or {}) for p in response.points]

    def get_alias_target(self, alias: str) -> str | None:
        aliases = self.client.get_aliases().aliases
        for item in aliases:
            if item.alias_name == alias:
                return item.collection_name
        return None

    def switch_alias(self, alias: str, new_collection: str) -> None:
        current = self.get_alias_target(alias)
        ops: list[UpdateCollectionAliasesOperation] = []
        if current:
            ops.append(UpdateCollectionAliasesOperation(delete_alias=DeleteAlias(alias_name=alias)))
        ops.append(UpdateCollectionAliasesOperation(create_alias=CreateAlias(alias_name=alias, collection_name=new_collection)))
        self.client.update_collection_aliases(change_aliases_operations=ops)

    def ensure_alias(self, alias: str, collection: str) -> None:
        target = self.get_alias_target(alias)
        if target is None:
            self.client.update_collection_aliases(
                change_aliases_operations=[
                    UpdateCollectionAliasesOperation(create_alias=CreateAlias(alias_name=alias, collection_name=collection)),
                ]
            )

    def count(self, collection: str) -> int:
        return self.client.count(collection_name=collection, exact=True).count
