from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from src.config.settings import Settings
from src.ingestion.parsers import load_document
from src.ingestion.policy import build_doc_id, resolve_document_policy


class IngestionPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunking.chunk_size_tokens,
            chunk_overlap=settings.chunking.chunk_overlap_tokens,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=lambda x: len(tokenizer.encode(x, add_special_tokens=False)),
        )
        self.embedder = SentenceTransformer(
            settings.retrieval.embedding_model_name,
            cache_folder="models",
            device="cuda",
        )

    def collect_documents(self) -> list[Path]:
        root = Path(self.settings.document_root)
        return [p for p in root.rglob("*") if p.suffix.lower() in {".pdf", ".docx"}]

    def build_points(self, files: list[Path], collection_version: str) -> tuple[list[PointStruct], dict[str, Any]]:
        report: dict[str, Any] = {
            "files_found": len(files),
            "files_processed": 0,
            "files_skipped": 0,
            "files_failed": 0,
            "parse_errors": [],
            "chunks_created": 0,
        }
        points: list[PointStruct] = []
        texts: list[str] = []
        payloads: list[dict[str, Any]] = []

        for file_path in files:
            try:
                policy = resolve_document_policy(self.settings, file_path)
                docs = load_document(str(file_path))
                if not docs:
                    report["files_skipped"] += 1
                    continue
                doc_id = build_doc_id(file_path)
                for d in docs:
                    chunks = self.splitter.split_text(d["text"])
                    for idx, ch in enumerate(chunks):
                        if len(ch.strip()) < self.settings.chunking.min_chunk_chars:
                            continue
                        texts.append(ch)
                        payloads.append(
                            {
                                "text": ch,
                                "doc_id": doc_id,
                                "source_path": str(file_path),
                                "domain": policy["domain"],
                                "allowed_groups": policy["allowed_groups"],
                                "status": policy["status"],
                                "chunk_index": idx,
                                "index_version": collection_version,
                                "content_hash": hashlib.sha256(ch.encode()).hexdigest(),
                                "page_number": d["metadata"].get("page_number"),
                            }
                        )
                report["files_processed"] += 1
            except Exception as exc:
                report["files_failed"] += 1
                report["parse_errors"].append({"file": str(file_path), "error": str(exc)})

        if texts:
            vectors = self.embedder.encode(texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
            for i, (vec, payload) in enumerate(zip(vectors, payloads)):
                points.append(PointStruct(id=i + 1, vector=vec.tolist(), payload=payload))
        report["chunks_created"] = len(points)
        return points, report
