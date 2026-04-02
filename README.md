# RAG Project

## Description
RAG pipeline for processing documents using vector search and LLM.

## Structure
- ingestion/ — document parsing and indexing
- retrieval/ — vector search and reranking
- generation/ — LLM interaction
- app.py — entrypoint

## Requirements
pip install -r requirements.txt

## Notes
- models/ is not included (local models)
- qdrant_storage/ is not included (vector DB state)

## TODO
- dockerization
- API layer
- reproducible ingestion pipeline
