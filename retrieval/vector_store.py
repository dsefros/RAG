# retrieval/vector_store.py
import logging
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QdrantVectorStore:
    def __init__(
        self,
        collection_name: str = "rag_product",  # "rag_product" или "rag_regulatory"
        embedding_model_name: str = "BAAI/bge-m3",
        host: str = "localhost",
        port: int = 6333,
        cache_folder: str = "models"
    ):
        self.collection_name = collection_name
        self.client = QdrantClient(host=host, port=port)
        
        logger.info(f"Загрузка модели эмбеддингов: {embedding_model_name} для коллекции {collection_name}")
        self.embedding_model = SentenceTransformer(
            embedding_model_name,
            cache_folder=cache_folder,
            device="cuda"
        )
        self._create_collection()

    def _create_collection(self):
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
            )
            logger.info(f"Коллекция создана: {self.collection_name}")
        else:
            logger.info(f"Коллекция уже существует: {self.collection_name}")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        logger.info(f"Генерация эмбеддингов для {len(texts)} текстов в {self.collection_name}...")
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings.tolist()

    def add_documents(self, documents: List[Dict[str, Any]], replace: bool = True, batch_size: int = 100):
        if not documents:
            logger.warning("Нет документов для добавления")
            return

        if replace:
            if self.client.collection_exists(self.collection_name):
                logger.info(f"Удаление старой коллекции: {self.collection_name}")
                self.client.delete_collection(self.collection_name)
            self._create_collection()
        
        texts = [doc["text"] for doc in documents]
        embeddings = self.embed_texts(texts)
        
        import hashlib
        points = []
        for doc, embedding in zip(documents, embeddings):
            doc_id = hashlib.md5(f"{doc['text']}{doc['metadata']['source']}".encode()).hexdigest()
            points.append(
                PointStruct(
                    id=doc_id,
                    vector=embedding,
                    payload={
                        "text": doc["text"],
                        "metadata": {**doc["metadata"], "source_db": self.collection_name}
                    }
                )
            )
        
        total = len(points)
        logger.info(f"Добавление {total} точек в {self.collection_name} пакетами по {batch_size}...")
        for i in range(0, total, batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(collection_name=self.collection_name, points=batch)
            logger.info(f"  [{self.collection_name}] Отправлено: {min(i + batch_size, total)}/{total}")
        logger.info(f"✅ Документы успешно добавлены в {self.collection_name}")

    def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        if not isinstance(query, str) or not query.strip():
            logger.warning("Получен пустой или некорректный запрос")
            return []
        
        clean_query = query.strip()
        try:
            query_embedding = self.embedding_model.encode(
                [clean_query],
                convert_to_numpy=True,
                normalize_embeddings=True
            )[0].tolist()
        except Exception as e:
            logger.error(f"Ошибка при генерации эмбеддинга для '{clean_query}': {e}")
            return []
        
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k
        )
        return [{
            "text": hit.payload["text"],
            "metadata": hit.payload["metadata"],
            "score": hit.score,
            "source_db": self.collection_name  # Сохраняем источник
        } for hit in results.points]