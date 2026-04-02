import os
from pathlib import Path
from typing import List, Dict, Any
import logging
from pypdf import PdfReader
from docx import Document

logger = logging.getLogger(__name__)

def load_pdf(file_path: str) -> List[Dict[str, Any]]:
    """Загружает PDF файл с помощью pypdf (без сложных зависимостей)."""
    logger.info(f"Загрузка PDF: {file_path}")
    reader = PdfReader(file_path)
    pages = []
    
    for page_num, page in enumerate(reader.pages, 1):
        text = page.extract_text()
        if text.strip():
            pages.append({
                "text": text,
                "metadata": {
                    "source": os.path.basename(file_path),
                    "source_type": "pdf",
                    "page_number": page_num
                }
            })
    
    return pages

def load_docx(file_path: str) -> List[Dict[str, Any]]:
    """Загружает DOCX файл."""
    logger.info(f"Загрузка DOCX: {file_path}")
    doc = Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        if para.text.strip():
            full_text.append(para.text)
    
    text = "\n".join(full_text)
    if not text.strip():
        return []
    
    return [{
        "text": text,
        "metadata": {
            "source": os.path.basename(file_path),
            "source_type": "docx",
            "page_number": 1
        }
    }]

def load_document(file_path: str) -> List[Dict[str, Any]]:
    """Универсальный загрузчик документов."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}")
    
    if file_path.suffix.lower() == '.pdf':
        return load_pdf(str(file_path))
    elif file_path.suffix.lower() == '.docx':
        return load_docx(str(file_path))
    else:
        raise ValueError(f"Неподдерживаемый формат: {file_path.suffix}")