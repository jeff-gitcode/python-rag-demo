# Simple RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a simple Q&A RAG system with clean architecture that runs locally

**Architecture:** Clean architecture with domain, application, infrastructure, and presentation layers. Domain has zero dependencies, infrastructure implements domain interfaces.

**Tech Stack:** Python, langchain, sentence-transformers, chromadb, ollama

---

## File Structure

```
src/
├── domain/
│   ├── entities/
│   │   ├── __init__.py
│   │   ├── document.py
│   │   ├── chunk.py
│   │   ├── query.py
│   │   └── answer.py
│   └── repositories/
│       ├── __init__.py
│       ├── document_repository.py
│       ├── embedding_service.py
│       ├── llm_service.py
│       └── document_loader.py
├── application/
│   ├── __init__.py
│   ├── ingest_document.py
│   └── query_documents.py
├── infrastructure/
│   ├── __init__.py
│   ├── embedding/
│   │   ├── __init__.py
│   │   └── sentence_transformers_embedding.py
│   ├── vectorstore/
│   │   ├── __init__.py
│   │   └── chromadb_store.py
│   ├── llm/
│   │   ├── __init__.py
│   │   └── ollama_llm.py
│   └── loader/
│       ├── __init__.py
│       └── text_file_loader.py
├── presentation/
│   ├── __init__.py
│   └── main.py
└── __init__.py
tests/
├── domain/
│   ├── entities/
│   │   └── test_entities.py
│   └── repositories/
│       └── test_interfaces.py
├── application/
│   └── test_use_cases.py
└── infrastructure/
    └── test_implementations.py
requirements.txt
```

---

### Task 1: Domain Layer - Entities

**Files:**
- Create: `src/domain/entities/__init__.py`
- Create: `src/domain/entities/document.py`
- Create: `src/domain/entities/chunk.py`
- Create: `src/domain/entities/query.py`
- Create: `src/domain/entities/answer.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/domain/entities/test_entities.py
import pytest
from src.domain.entities.document import Document
from src.domain.entities.chunk import Chunk
from src.domain.entities.query import Query
from src.domain.entities.answer import Answer


def test_document_creation():
    doc = Document(
        id="doc-1",
        content="Sample document content",
        source_path="/path/to/doc.txt"
    )
    assert doc.id == "doc-1"
    assert doc.content == "Sample document content"
    assert doc.source_path == "/path/to/doc.txt"
    assert doc.created_at is not None


def test_chunk_creation():
    chunk = Chunk(
        id="chunk-1",
        document_id="doc-1",
        content="Sample chunk content"
    )
    assert chunk.id == "chunk-1"
    assert chunk.document_id == "doc-1"
    assert chunk.content == "Sample chunk content"
    assert chunk.embedding is None


def test_query_creation():
    query = Query(
        id="query-1",
        text="What is this about?"
    )
    assert query.id == "query-1"
    assert query.text == "What is this about?"
    assert query.embedding is None


def test_answer_creation():
    answer = Answer(
        id="answer-1",
        query_id="query-1",
        content="This is the answer",
        sources=["doc-1"]
    )
    assert answer.id == "answer-1"
    assert answer.query_id == "query-1"
    assert answer.content == "This is the answer"
    assert answer.sources == ["doc-1"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/domain/entities/test_entities.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'src'"

- [ ] **Step 3: Write minimal implementation**

```python
# src/domain/entities/document.py
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Document:
    id: str
    content: str
    source_path: str
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
```

```python
# src/domain/entities/chunk.py
from dataclasses import dataclass
from typing import Optional


@dataclass
class Chunk:
    id: str
    document_id: str
    content: str
    embedding: Optional[list[float]] = None
```

```python
# src/domain/entities/query.py
from dataclasses import dataclass
from typing import Optional


@dataclass
class Query:
    id: str
    text: str
    embedding: Optional[list[float]] = None
```

```python
# src/domain/entities/answer.py
from dataclasses import dataclass
from typing import Optional


@dataclass
class Answer:
    id: str
    query_id: str
    content: str
    sources: list[str]
```

```python
# src/domain/entities/__init__.py
from src.domain.entities.document import Document
from src.domain.entities.chunk import Chunk
from src.domain.entities.query import Query
from src.domain.entities.answer import Answer

__all__ = ["Document", "Chunk", "Query", "Answer"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/domain/entities/test_entities.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/domain/entities/ tests/domain/entities/
git commit -m "feat: add domain entities (Document, Chunk, Query, Answer)"
```

---

### Task 2: Domain Layer - Repository Interfaces

**Files:**
- Create: `src/domain/repositories/__init__.py`
- Create: `src/domain/repositories/document_repository.py`
- Create: `src/domain/repositories/embedding_service.py`
- Create: `src/domain/repositories/llm_service.py`
- Create: `src/domain/repositories/document_loader.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/domain/repositories/test_interfaces.py
import pytest
from abc import ABC
from src.domain.repositories.document_repository import DocumentRepository
from src.domain.repositories.embedding_service import EmbeddingService
from src.domain.repositories.llm_service import LLMService
from src.domain.repositories.document_loader import DocumentLoader


def test_document_repository_is_abc():
    assert issubclass(DocumentRepository, ABC)


def test_embedding_service_is_abc():
    assert issubclass(EmbeddingService, ABC)


def test_llm_service_is_abc():
    assert issubclass(LLMService, ABC)


def test_document_loader_is_abc():
    assert issubclass(DocumentLoader, ABC)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/domain/repositories/test_interfaces.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Write minimal implementation**

```python
# src/domain/repositories/document_repository.py
from abc import ABC, abstractmethod
from typing import Optional
from src.domain.entities.chunk import Chunk


class DocumentRepository(ABC):
    @abstractmethod
    def add_chunk(self, chunk: Chunk) -> None:
        pass

    @abstractmethod
    def search(self, embedding: list[float], top_k: int) -> list[Chunk]:
        pass

    @abstractmethod
    def get_all_documents(self) -> list[str]:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass
```

```python
# src/domain/repositories/embedding_service.py
from abc import ABC, abstractmethod


class EmbeddingService(ABC):
    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        pass

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        pass
```

```python
# src/domain/repositories/llm_service.py
from abc import ABC, abstractmethod
from typing import Optional


class LLMService(ABC):
    @abstractmethod
    def generate(self, prompt: str, context: str) -> str:
        pass
```

```python
# src/domain/repositories/document_loader.py
from abc import ABC, abstractmethod
from src.domain.entities.document import Document


class DocumentLoader(ABC):
    @abstractmethod
    def load(self, path: str) -> Document:
        pass
```

```python
# src/domain/repositories/__init__.py
from src.domain.repositories.document_repository import DocumentRepository
from src.domain.repositories.embedding_service import EmbeddingService
from src.domain.repositories.llm_service import LLMService
from src.domain.repositories.document_loader import DocumentLoader

__all__ = ["DocumentRepository", "EmbeddingService", "LLMService", "DocumentLoader"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/domain/repositories/test_interfaces.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/domain/repositories/ tests/domain/repositories/
git commit -m "feat: add domain repository interfaces"
```

---

### Task 3: Infrastructure - Embedding Service

**Files:**
- Create: `src/infrastructure/embedding/__init__.py`
- Create: `src/infrastructure/embedding/sentence_transformers_embedding.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/infrastructure/test_implementations.py
import pytest
from src.infrastructure.embedding.sentence_transformers_embedding import SentenceTransformersEmbedding


def test_embedding_service_embed_text():
    service = SentenceTransformersEmbedding()
    embedding = service.embed_text("Hello world")
    assert isinstance(embedding, list)
    assert len(embedding) > 0
    assert all(isinstance(x, float) for x in embedding)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/infrastructure/test_implementations.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Write minimal implementation**

```python
# src/infrastructure/embedding/sentence_transformers_embedding.py
from sentence_transformers import SentenceTransformer
from src.domain.repositories.embedding_service import EmbeddingService


class SentenceTransformersEmbedding(EmbeddingService):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text: str) -> list[float]:
        embedding = self.model.encode(text)
        return embedding.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
```

```python
# src/infrastructure/embedding/__init__.py
from src.infrastructure.embedding.sentence_transformers_embedding import SentenceTransformersEmbedding

__all__ = ["SentenceTransformersEmbedding"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/infrastructure/test_implementations.py::test_embedding_service_embed_text -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/infrastructure/embedding/ tests/infrastructure/
git commit -m "feat: add SentenceTransformers embedding implementation"
```

---

### Task 4: Infrastructure - ChromaDB Vector Store

**Files:**
- Create: `src/infrastructure/vectorstore/__init__.py`
- Create: `src/infrastructure/vectorstore/chromadb_store.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/infrastructure/test_implementations.py (add to existing)
import pytest
from src.infrastructure.vectorstore.chromadb_store import ChromaDBStore
from src.domain.entities.chunk import Chunk


def test_chromadb_store_add_and_search():
    store = ChromaDBStore()
    chunk = Chunk(
        id="chunk-1",
        document_id="doc-1",
        content="The quick brown fox",
        embedding=[0.1] * 384
    )
    store.add_chunk(chunk)
    results = store.search([0.1] * 384, top_k=1)
    assert len(results) == 1
    assert results[0].id == "chunk-1"
    store.clear()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/infrastructure/test_implementations.py::test_chromadb_store_add_and_search -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Write minimal implementation**

```python
# src/infrastructure/vectorstore/chromadb_store.py
import chromadb
from chromadb.config import Settings
from src.domain.repositories.document_repository import DocumentRepository
from src.domain.entities.chunk import Chunk


class ChromaDBStore(DocumentRepository):
    def __init__(self):
        self.client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.collection = self.client.create_collection("documents")

    def add_chunk(self, chunk: Chunk) -> None:
        self.collection.add(
            ids=[chunk.id],
            documents=[chunk.content],
            metadatas=[{"document_id": chunk.document_id}]
        )

    def search(self, embedding: list[float], top_k: int) -> list[Chunk]:
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k
        )
        chunks = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                chunks.append(Chunk(
                    id=chunk_id,
                    document_id=results["metadatas"][0][i]["document_id"],
                    content=results["documents"][0][i]
                ))
        return chunks

    def get_all_documents(self) -> list[str]:
        return self.collection.get()["ids"]

    def clear(self) -> None:
        self.client.delete_collection("documents")
        self.collection = self.client.create_collection("documents")
```

```python
# src/infrastructure/vectorstore/__init__.py
from src.infrastructure.vectorstore.chromadb_store import ChromaDBStore

__all__ = ["ChromaDBStore"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/infrastructure/test_implementations.py::test_chromadb_store_add_and_search -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/infrastructure/vectorstore/ tests/infrastructure/
git commit -m "feat: add ChromaDB vector store implementation"
```

---

### Task 5: Infrastructure - Ollama LLM

**Files:**
- Create: `src/infrastructure/llm/__init__.py`
- Create: `src/infrastructure/llm/ollama_llm.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/infrastructure/test_implementations.py (add to existing)
import pytest
from unittest.mock import Mock, patch
from src.infrastructure.llm.ollama_llm import OllamaLLM


def test_ollama_llm_generate():
    with patch("src.infrastructure.llm.ollama_llm.ollama") as mock_ollama:
        mock_ollama.chat.return_value = {"message": {"content": "Test response"}}
        llm = OllamaLLM()
        result = llm.generate("Question?", "Context")
        assert result == "Test response"
        mock_ollama.chat.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/infrastructure/test_implementations.py::test_ollama_llm_generate -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Write minimal implementation**

```python
# src/infrastructure/llm/ollama_llm.py
import ollama
from src.domain.repositories.llm_service import LLMService


class OllamaLLM(LLMService):
    def __init__(self, model: str = "llama3.2"):
        self.model = model

    def generate(self, prompt: str, context: str) -> str:
        full_prompt = f"""Context: {context}

Question: {prompt}

Answer:"""

        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": full_prompt}]
        )
        return response["message"]["content"]
```

```python
# src/infrastructure/llm/__init__.py
from src.infrastructure.llm.ollama_llm import OllamaLLM

__all__ = ["OllamaLLM"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/infrastructure/test_implementations.py::test_ollama_llm_generate -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/infrastructure/llm/ tests/infrastructure/
git commit -m "feat: add Ollama LLM implementation"
```

---

### Task 6: Infrastructure - Document Loader

**Files:**
- Create: `src/infrastructure/loader/__init__.py`
- Create: `src/infrastructure/loader/text_file_loader.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/infrastructure/test_implementations.py (add to existing)
import pytest
import tempfile
import os
from src.infrastructure.loader.text_file_loader import TextFileLoader


def test_text_file_loader():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("Test content")
        temp_path = f.name
    
    try:
        loader = TextFileLoader()
        doc = loader.load(temp_path)
        assert doc.content == "Test content"
        assert doc.source_path == temp_path
    finally:
        os.unlink(temp_path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/infrastructure/test_implementations.py::test_text_file_loader -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Write minimal implementation**

```python
# src/infrastructure/loader/text_file_loader.py
import uuid
from src.domain.repositories.document_loader import DocumentLoader
from src.domain.entities.document import Document


class TextFileLoader(DocumentLoader):
    def load(self, path: str) -> Document:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return Document(
            id=str(uuid.uuid4()),
            content=content,
            source_path=path
        )
```

```python
# src/infrastructure/loader/__init__.py
from src.infrastructure.loader.text_file_loader import TextFileLoader

__all__ = ["TextFileLoader"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/infrastructure/test_implementations.py::test_text_file_loader -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/infrastructure/loader/ tests/infrastructure/
git commit -m "feat: add text file loader implementation"
```

---

### Task 7: Application Layer - Use Cases

**Files:**
- Create: `src/application/__init__.py`
- Create: `src/application/ingest_document.py`
- Create: `src/application/query_documents.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/application/test_use_cases.py
import pytest
from unittest.mock import Mock
from src.application.ingest_document import IngestDocumentUseCase
from src.application.query_documents import QueryDocumentsUseCase


def test_ingest_use_case():
    mock_loader = Mock()
    mock_embedding = Mock()
    mock_repo = Mock()
    
    mock_loader.load.return_value = Mock(
        id="doc-1",
        content="Sample content here"
    )
    mock_embedding.embed_documents.return_value = [[0.1] * 384]
    
    use_case = IngestDocumentUseCase(
        loader=mock_loader,
        embedding_service=mock_embedding,
        repository=mock_repo
    )
    
    use_case.execute("/path/to/doc.txt")
    
    mock_loader.load.assert_called_once_with("/path/to/doc.txt")
    mock_embedding.embed_documents.assert_called_once()
    mock_repo.add_chunk.assert_called_once()


def test_query_use_case():
    mock_embedding = Mock()
    mock_repo = Mock()
    mock_llm = Mock()
    
    mock_embedding.embed_text.return_value = [0.1] * 384
    mock_repo.search.return_value = [Mock(content="Relevant chunk")]
    mock_llm.generate.return_value = "Answer from LLM"
    
    use_case = QueryDocumentsUseCase(
        embedding_service=mock_embedding,
        repository=mock_repo,
        llm_service=mock_llm
    )
    
    result = use_case.execute("What is this?", top_k=3)
    
    assert result.content == "Answer from LLM"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/application/test_use_cases.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Write minimal implementation**

```python
# src/application/ingest_document.py
import uuid
from src.domain.repositories.document_loader import DocumentLoader
from src.domain.repositories.embedding_service import EmbeddingService
from src.domain.repositories.document_repository import DocumentRepository
from src.domain.entities.chunk import Chunk


class IngestDocumentUseCase:
    def __init__(
        self,
        loader: DocumentLoader,
        embedding_service: EmbeddingService,
        repository: DocumentRepository
    ):
        self.loader = loader
        self.embedding_service = embedding_service
        self.repository = repository

    def execute(self, path: str, chunk_size: int = 500):
        document = self.loader.load(path)
        
        chunks = self._chunk_text(document.content, chunk_size)
        
        embeddings = self.embedding_service.embed_documents(chunks)
        
        for i, chunk_text in enumerate(chunks):
            chunk = Chunk(
                id=str(uuid.uuid4()),
                document_id=document.id,
                content=chunk_text,
                embedding=embeddings[i]
            )
            self.repository.add_chunk(chunk)

    def _chunk_text(self, text: str, chunk_size: int) -> list[str]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks if chunks else [text]
```

```python
# src/application/query_documents.py
import uuid
from src.domain.repositories.embedding_service import EmbeddingService
from src.domain.repositories.document_repository import DocumentRepository
from src.domain.repositories.llm_service import LLMService
from src.domain.entities.query import Query
from src.domain.entities.answer import Answer


class QueryDocumentsUseCase:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        repository: DocumentRepository,
        llm_service: LLMService
    ):
        self.embedding_service = embedding_service
        self.repository = repository
        self.llm_service = llm_service

    def execute(self, question: str, top_k: int = 3) -> Answer:
        query = Query(id=str(uuid.uuid4()), text=question)
        
        query.embedding = self.embedding_service.embed_text(question)
        
        relevant_chunks = self.repository.search(query.embedding, top_k)
        
        context = "\n\n".join([chunk.content for chunk in relevant_chunks])
        
        answer_text = self.llm_service.generate(question, context)
        
        sources = [chunk.document_id for chunk in relevant_chunks]
        
        return Answer(
            id=str(uuid.uuid4()),
            query_id=query.id,
            content=answer_text,
            sources=sources
        )
```

```python
# src/application/__init__.py
from src.application.ingest_document import IngestDocumentUseCase
from src.application.query_documents import QueryDocumentsUseCase

__all__ = ["IngestDocumentUseCase", "QueryDocumentsUseCase"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/application/test_use_cases.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/application/ tests/application/
git commit -m "feat: add application use cases"
```

---

### Task 8: Presentation Layer - CLI

**Files:**
- Create: `src/presentation/__init__.py`
- Create: `src/presentation/main.py`
- Create: `src/__init__.py`
- Create: `src/main.py`

- [ ] **Step 1: Write the failing test**

```python
# No separate test - we'll test CLI manually
```

- [ ] **Step 2: Run test to verify it fails**

Skip - CLI will be tested manually

- [ ] **Step 3: Write minimal implementation**

```python
# src/presentation/main.py
import argparse
from src.infrastructure.loader.text_file_loader import TextFileLoader
from src.infrastructure.embedding.sentence_transformers_embedding import SentenceTransformersEmbedding
from src.infrastructure.vectorstore.chromadb_store import ChromaDBStore
from src.infrastructure.llm.ollama_llm import OllamaLLM
from src.application.ingest_document import IngestDocumentUseCase
from src.application.query_documents import QueryDocumentsUseCase


def main():
    parser = argparse.ArgumentParser(description="Simple RAG CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents")
    ingest_parser.add_argument("path", help="Path to document or directory")
    
    query_parser = subparsers.add_parser("query", help="Query documents")
    query_parser.add_argument("question", help="Question to ask")
    query_parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve")
    
    args = parser.parse_args()
    
    loader = TextFileLoader()
    embedding = SentenceTransformersEmbedding()
    vector_store = ChromaDBStore()
    llm = OllamaLLM()
    
    if args.command == "ingest":
        use_case = IngestDocumentUseCase(
            loader=loader,
            embedding_service=embedding,
            repository=vector_store
        )
        use_case.execute(args.path)
        print(f"Successfully ingested: {args.path}")
    
    elif args.command == "query":
        use_case = QueryDocumentsUseCase(
            embedding_service=embedding,
            repository=vector_store,
            llm_service=llm
        )
        answer = use_case.execute(args.question, args.top_k)
        print(f"\nAnswer: {answer.content}\n")
        print(f"Sources: {answer.sources}")


if __name__ == "__main__":
    main()
```

```python
# src/presentation/__init__.py
from src.presentation.main import main

__all__ = ["main"]
```

```python
# src/__init__.py
```

```python
# src/main.py
from src.presentation.main import main

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it works**

Run: `python -m src.main --help`
Expected: Shows help output

- [ ] **Step 5: Commit**

```bash
git add src/presentation/ src/main.py src/__init__.py
git commit -m "feat: add CLI presentation layer"
```

---

### Task 9: Requirements and Final Integration

**Files:**
- Create: `requirements.txt`

- [ ] **Step 1: Write requirements.txt**

```
langchain
langchain-community
sentence-transformers
chromadb
ollama
pytest
```

- [ ] **Step 2: Install and test**

```bash
pip install -r requirements.txt
python -m src.main --help
```

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "chore: add requirements.txt"
```

---

**Plan complete.** Two execution options:

1. **Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks
2. **Inline Execution** - Execute tasks in this session using executing-plans

Which approach?
