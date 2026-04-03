from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from docx import Document
from pypdf import PdfReader


def load_pdf(file_path: str) -> list[dict[str, Any]]:
    reader = PdfReader(file_path)
    pages: list[dict[str, Any]] = []
    for page_num, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(
                {
                    "text": text,
                    "metadata": {
                        "source": os.path.basename(file_path),
                        "source_type": "pdf",
                        "page_number": page_num,
                    },
                }
            )
    return pages


def load_docx(file_path: str) -> list[dict[str, Any]]:
    doc = Document(file_path)
    text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    if not text.strip():
        return []
    return [
        {
            "text": text,
            "metadata": {
                "source": os.path.basename(file_path),
                "source_type": "docx",
                "page_number": 1,
            },
        }
    ]


def load_document(file_path: str) -> list[dict[str, Any]]:
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(str(path))
    if suffix == ".docx":
        return load_docx(str(path))
    raise ValueError(f"Unsupported file type: {suffix}")
