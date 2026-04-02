# test_ingestion.py
from ingestion.pipeline import DocumentIngestionPipeline
from retrieval.vector_store import QdrantVectorStore

def main():
    print("=== Этап 1: Ingestion продуктовых гайдов ===")
    pipeline_product = DocumentIngestionPipeline(chunk_size=300, chunk_overlap=50)  # Токены!
    product_chunks = pipeline_product.run("data/product_guides")
    
    print(f"Обработано {len(product_chunks)} продуктовых чанков")
    
    print("\n=== Этап 2: Ingestion регуляторных документов ===")
    pipeline_regulatory = DocumentIngestionPipeline(chunk_size=500, chunk_overlap=80)  # Больше для юр.текстов
    regulatory_chunks = pipeline_regulatory.run("data/regulatory_docs")
    
    print(f"Обработано {len(regulatory_chunks)} регуляторных чанков")
    
    if not product_chunks and not regulatory_chunks:
        print("⚠️ Нет документов для индексации!")
        return
    
    print("\n=== Этап 3: Загрузка в Qdrant ===")
    if product_chunks:
        print("-> Загрузка в rag_product")
        product_store = QdrantVectorStore(collection_name="rag_product")
        product_store.add_documents(product_chunks, replace=True)
    
    if regulatory_chunks:
        print("-> Загрузка в rag_regulatory")
        regulatory_store = QdrantVectorStore(collection_name="rag_regulatory")
        regulatory_store.add_documents(regulatory_chunks, replace=True)
    
    print("\n✅ Обе базы успешно обновлены!")

if __name__ == "__main__":
    main()