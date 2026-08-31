````markdown
# Technical Documentation RAG Chatbot

A retrieval-augmented generation application for answering questions from technical documentation.

## Architecture

```text
Documentation
      ↓
   Chunking
      ↓
  Embeddings
      ↓
Vector representations
      ↓

User question
      ↓
Question embedding
      ↓
Similarity search
      ↓
Relevant chunks
      ↓
OpenAI
      ↓
Grounded answer
````

## Core concepts

### 1. Chunking

Large documents are split into smaller pieces.

### 2. Embeddings

Each chunk is converted into a numerical vector representing its semantic meaning.

### 3. Retrieval

The user's question is also converted into a vector.

The application compares the question vector with the document vectors.

The most similar chunks become the context.

### 4. Generation

The retrieved chunks are sent to the OpenAI model.

The model generates an answer based on those chunks.

## Why RAG?

A normal chatbot might answer:

> "I know how VPNs work."

A RAG chatbot instead works like:

> "Here are the relevant sections from your company's documentation. Answer using these."

This reduces unsupported answers and allows an organization to use its own documentation.

## Real-world applications

This architecture can become:

* IT Service Desk assistant
* Company knowledge assistant
* Developer documentation assistant
* HR policy assistant
* Internal support chatbot
* Product documentation chatbot
* Training assistant

## Current limitation

This demo keeps embeddings in memory.

A production application should use a vector database such as ChromaDB, Pinecone, pgvector or another suitable vector store.

## Run

```powershell
python app.py
```
