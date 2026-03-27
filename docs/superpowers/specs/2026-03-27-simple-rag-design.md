# Simple Q&A RAG - Design

## Overview
A simple Retrieval-Augmented Generation (RAG) system for Q&A over local documents. Runs entirely locally using open-source tools.

## Technology Stack
- **Embedding**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector Store**: ChromaDB (in-memory)
- **LLM**: Ollama (local)
- **Framework**: langchain

## Architecture (Clean Architecture)

```
src/
├── domain/                    # Core business logic (no external deps)
│   ├── entities/
│   │   ├── document.py        # Document entity
│   │   ├── chunk.py           # Chunk entity
│   │   ├── query.py           # Query entity
│   │   └── answer.py          # Answer entity
│   └── repositories/
│       ├── document_repository.py    # Abstract interface
│       ├── embedding_service.py       # Abstract interface
│       ├── llm_service.py             # Abstract interface
│       └── document_loader.py         # Abstract interface
│
├── application/               # Use cases
│   ├── ingest_document.py
│   └── query_documents.py
│
├── infrastructure/           # External implementations
│   ├── embedding/
│   │   └── sentence_transformers_embedding.py
│   ├── vectorstore/
│   │   └── chromadb_store.py
│   ├── llm/
│   │   └── ollama_llm.py
│   └── loader/
│       └── text_file_loader.py
│
└── presentation/              # CLI
    └── main.py
```

## Dependency Rule
- Domain: Zero dependencies
- Application: Depends only on Domain
- Infrastructure: Implements Domain interfaces
- Presentation: Depends on Application

## Components

### Domain Layer
- **Document**: id, content, source_path, created_at
- **Chunk**: id, document_id, content, embedding
- **Query**: id, text, embedding
- **Answer**: id, query_id, content, sources

### Application Layer
- **IngestDocumentUseCase**: Load document → chunk → embed → store
- **QueryDocumentsUseCase**: Embed query → search → generate answer

### Infrastructure Layer
- **SentenceTransformersEmbedding**: Implements EmbeddingService
- **ChromaDBStore**: Implements DocumentRepository
- **OllamaLLM**: Implements LLMService
- **TextFileLoader**: Implements DocumentLoader

## Data Flow

### Ingest Flow
```
File → TextFileLoader → Document → chunk → SentenceTransformersEmbedding → ChromaDBStore
```

### Query Flow
```
Query → SentenceTransformersEmbedding → ChromaDBStore (search) → top-k chunks → OllamaLLM → Answer
```

## Acceptance Criteria
1. CLI accepts `ingest <path>` to index documents
2. CLI accepts `query <question>` to ask questions
3. Uses sentence-transformers for embeddings
4. Uses ChromaDB for vector storage
5. Uses Ollama for LLM generation
6. All code follows clean architecture layers
7. Dependencies only flow inward (toward domain)
