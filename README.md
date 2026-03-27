# Simple RAG

A simple Q&A RAG system built with clean architecture principles.

## Tech Stack

- **Embedding**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector Store**: ChromaDB (in-memory)
- **LLM**: Ollama (local)
- **Framework**: langchain

## Architecture

```
src/
├── domain/           # Core entities and interfaces
├── application/      # Use cases
├── infrastructure/    # External implementations
└── presentation/     # CLI
```

### Data Flow

```
                    ┌─────────────────┐
                    │  Text File      │
                    └────────┬────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │     TextFileLoader           │
              │   (DocumentLoader)           │
              └──────────────┬───────────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │   Document     │
                    └────────┬───────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │   IngestDocumentUseCase      │
              │   (chunk text)              │
              └──────────────┬───────────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │  SentenceTransformers       │
              │  (EmbeddingService)         │
              └──────────────┬───────────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │ ChromaDB Store │
                    │ (DocumentRepo) │
                    └────────┬───────┘
                             │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
         ▼                    ▼                    ▼
  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
  │  Query CLI  │      │  Search     │      │  Ollama    │
  └──────┬──────┘      └──────┬──────┘      │  (LLM)     │
         │                   │             └──────┬──────┘
         ▼                   │                    │
┌─────────────────┐          │                    │
│ SentenceTrans- │◄─────────┘                    │
│ formers        │                               │
└────────┬────────┘                               │
         │                                        │
         ▼                                        ▼
┌─────────────────┐                    ┌────────────────────┐
│ ChromaDB Search │                    │     Answer         │
│ (top-k chunks) │───────────────────►│                    │
└─────────────────┘                    └────────────────────┘
```

## Setup

```bash
pip install -r requirements.txt
```

Make sure [Ollama](https://ollama.ai) is installed and running with a model:

```bash
ollama pull llama3.2
```

## Usage

### Ingest documents

```bash
python -m src.main ingest path/to/document.txt
```

### Query

```bash
python -m src.main query "Your question here"
```

## Testing

```bash
PYTHONPATH=. pytest
```
