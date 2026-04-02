# app.py
from typing import List, Dict, Any
import logging
from retrieval.vector_store import QdrantVectorStore
from retrieval.reranker import Reranker
from generation.llm_client import LLMClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def get_hybrid_context(
    query: str,
    product_store: QdrantVectorStore,
    regulatory_store: QdrantVectorStore,
    reranker: Reranker
) -> List[Dict[str, Any]]:
    """Получает 5 релевантных чанков из продуктовой базы и 3 из регуляторной"""
    
    # Создаём логгер внутри функции
    import logging
    logger = logging.getLogger(__name__)
    
    # Поиск в обеих базах
    logger.info("Поиск в продуктовой базе...")
    product_results = product_store.search(query, top_k=20)
    logger.info(f"Продуктовая база: найдено {len(product_results)} результатов")
    
    logger.info("Поиск в регуляторной базе...")
    regulatory_results = regulatory_store.search(query, top_k=20)
    logger.info(f"Регуляторная база: найдено {len(regulatory_results)} результатов")
    
    # Комбинируем результаты
    all_results = product_results + regulatory_results
    
    if not all_results:
        return []
    
    # Единый реранкинг для всех 40 чанков
    logger.info("Реранкинг всех результатов...")
    reranked = reranker.rerank(query, all_results, top_k=40)
    
    # Разделяем на базы
    product_reranked = [r for r in reranked if r["source_db"] == "rag_product"]
    regulatory_reranked = [r for r in reranked if r["source_db"] == "rag_regulatory"]
    
    # Берём 5 из продуктовой и 3 из регуляторной
    final_context = []
    
    # Продуктовые чанки (макс 5)
    product_limit = min(5, len(product_reranked))
    final_context.extend(product_reranked[:product_limit])
    
    # Регуляторные чанки (макс 3)
    regulatory_limit = min(3, len(regulatory_reranked))
    final_context.extend(regulatory_reranked[:regulatory_limit])
    
    # Если набрали меньше 8, добираем из оставшихся
    remaining_slots = 8 - len(final_context)
    if remaining_slots > 0:
        other_results = []
        if len(product_reranked) > product_limit:
            other_results.extend(product_reranked[product_limit:])
        if len(regulatory_reranked) > regulatory_limit:
            other_results.extend(regulatory_reranked[regulatory_limit:])
        
        # Сортируем по релевантности и добираем
        other_results.sort(key=lambda x: x["rerank_score"], reverse=True)
        final_context.extend(other_results[:remaining_slots])
    
    logger.info(f"Финальный контекст: {len(final_context)} чанков "
                f"({len([c for c in final_context if c['source_db']=='rag_product'])} продуктовых, "
                f"{len([c for c in final_context if c['source_db']=='rag_regulatory'])} регуляторных)")
    
    return final_context

def main():
    # Инициализация двух баз
    product_store = QdrantVectorStore(collection_name="rag_product")
    regulatory_store = QdrantVectorStore(collection_name="rag_regulatory")
    reranker = Reranker()
    llm = LLMClient()
    
    print("🚀 RAG-система с двумя базами знаний")
    print("• rag_product: инструкции по продуктам Sommers")
    print("• rag_regulatory: требования ЦБ, НСПК, EMV")
    print("Введите 'quit' для выхода\n")
    
    while True:
        query = input("Ваш вопрос: ").strip()
        if query.lower() in ['quit', 'exit', 'выход']:
            break
        if not query:
            continue
        
        # Получаем гибридный контекст 5+3
        context_docs = get_hybrid_context(
            query, 
            product_store, 
            regulatory_store, 
            reranker
        )
        
        if not context_docs:
            print("❌ Не найдено релевантных документов в обеих базах.\n")
            continue
        
        # Генерация ответа
        answer = llm.generate_answer(query, context_docs)
        print(f"\n💡 Ответ:\n{answer}\n")
        
        # Показываем источники с разделением по базам
        print("📄 Источники:")
        product_sources = [doc for doc in context_docs if doc["source_db"] == "rag_product"]
        regulatory_sources = [doc for doc in context_docs if doc["source_db"] == "rag_regulatory"]
        
        if product_sources:
            print("🔧 Продуктовые инструкции:")
            for i, doc in enumerate(product_sources, 1):
                source = doc['metadata']['source']
                page = doc['metadata'].get('page_number', 'N/A')
                print(f"   • {source} (стр. {page})")
        
        if regulatory_sources:
            print("\n📋 Регуляторные документы:")
            for i, doc in enumerate(regulatory_sources, 1):
                source = doc['metadata']['source']
                page = doc['metadata'].get('page_number', 'N/A')
                print(f"   • {source} (стр. {page})")
        print()

if __name__ == "__main__":
    main()