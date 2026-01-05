# Applied AI: Local RAG System for Document Question Answering

This project demonstrates an **Applied AI implementation of Retrieval-Augmented Generation (RAG)**, focusing on retrieval quality, system design tradeoffs, and debuggability rather than model novelty.

The system ingests PDF documents, converts them into semantically meaningful representations, retrieves relevant context using vector search, and generates grounded answers using a **local Large Language Model (LLM)**.

---

## Problem Statement

Large Language Models perform poorly on:
- private or domain-specific documents
- long-form content exceeding context limits
- tasks requiring source attribution

This project addresses these issues by augmenting generation with **retrieved, semantically relevant context**, enabling grounded and explainable answers.

---

## System Architecture

### Ingestion (Offline)
- Load PDF documents from disk
- Chunk text into semantically coherent nodes
- Generate dense vector embeddings for each chunk
- Store vectors in FAISS and persist text + metadata

### Inference (Online)

Embed user query
- Retrieve top-k relevant chunks via FAISS
- Augment prompt with retrieved context
- Generate an answer using a local LLM
- Expose source chunks and similarity scores

---

## Applied AI Focus Areas

- **Retrieval quality over model size**
- **Chunking strategy and overlap tuning**
- **Recall vs precision tradeoffs**
- **Source attribution and traceability**
- **Failure mode inspection and debugging**

---

## Key Features

- PDF ingestion and chunk-level indexing
- Semantic retrieval using FAISS
- Local LLM inference (no external APIs)
- Explicit source attribution for answers
- Retrieval debugging via node inspection

---

## Tech Stack

- **Python**
- **LlamaIndex**
- **FAISS**
- **Sentence-Transformers**
- **Ollama (local LLM runtime)**

---

## Project Structure

rag-pdf-chat/
├── ingest.py # Document ingestion and indexing
├── query.py # Retrieval and generation pipeline
├── requirements.txt
├── README.md
├── data/ # Input PDFs (gitignored)
├── storage/ # FAISS index + docstore (gitignored)
---
