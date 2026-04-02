# retrieval/reranker.py
import logging
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", cache_folder: str = "models"):
        logger.info(f"Загрузка реранкера: {model_name}")
        self.model = CrossEncoder(
            model_name,
            cache_folder=cache_folder,
            max_length=512
        )

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        if not documents:
            return []
        
        logger.info(f"Реранкинг {len(documents)} документов для запроса: {query}")
        pairs = [(query, doc["text"]) for doc in documents]
        
        # Уменьшаем batch_size для экономии VRAM
        scores = self.model.predict(pairs, batch_size=8)
        
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)
        
        reranked = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
        logger.info(f"Лучший score после реранкинга: {reranked[0]['rerank_score']:.4f}")
        return reranked[:top_k]