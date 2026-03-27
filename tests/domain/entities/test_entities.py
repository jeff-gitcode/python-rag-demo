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
