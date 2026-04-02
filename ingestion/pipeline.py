# ingestion/pipeline.py
import os
from pathlib import Path
from typing import List, Dict, Any
from ingestion.parsers import load_document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentIngestionPipeline:
    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50, doc_type: str = "product"):
        """
        chunk_size и chunk_overlap — в ТОКЕНАХ.
        doc_type: "product" или "regulatory"
        """
        # Для регуляторных документов используем большие чанки
        if doc_type == "regulatory":
            chunk_size = 500
            chunk_overlap = 80
        
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=lambda x: len(tokenizer.encode(x, add_special_tokens=False))
        )
        self.doc_type = doc_type

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        logger.info(f"Разбиение {len(documents)} документов ({self.doc_type}) на чанки...")
        all_chunks = []
        
        for doc in documents:
            texts = self.text_splitter.split_text(doc["text"])
            for i, text in enumerate(texts):
                if len(text.strip()) < 30:
                    continue
                all_chunks.append({
                    "text": text,
                    "metadata": {
                        **doc["metadata"],
                        "chunk_index": i,
                        "doc_type": self.doc_type
                    }
                })
        
        logger.info(f"Создано {len(all_chunks)} чанков ({self.doc_type})")
        return all_chunks

    def run(self, input_dir: str) -> List[Dict[str, Any]]:
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Директория не найдена: {input_dir}")
        
        supported_extensions = {'.pdf', '.docx'}
        all_docs = []
        
        for file_path in input_path.rglob('*'):
            if file_path.suffix.lower() in supported_extensions:
                try:
                    # Автоопределение типа документа по папке
                    doc_type = "product" if "product_guides" in str(file_path) else "regulatory"
                    docs = load_document(str(file_path))
                    # Добавляем тип в метаданные
                    for doc in docs:
                        doc["metadata"]["doc_type"] = doc_type
                    all_docs.extend(docs)
                except Exception as e:
                    logger.error(f"Ошибка при обработке {file_path}: {e}")
        
        chunks = self.chunk_documents(all_docs)
        return chunks