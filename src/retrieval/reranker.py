from __future__ import annotations

from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, model_name: str):
        self.model = CrossEncoder(model_name, cache_folder="models", max_length=512)

    def rerank(self, query: str, docs: list[dict], top_k: int) -> list[dict]:
        if not docs:
            return []
        pairs = [(query, d["text"]) for d in docs]
        scores = self.model.predict(pairs, batch_size=8)
        for d, s in zip(docs, scores):
            d["rerank_score"] = float(s)
        docs.sort(key=lambda x: x["rerank_score"], reverse=True)
        return docs[:top_k]
