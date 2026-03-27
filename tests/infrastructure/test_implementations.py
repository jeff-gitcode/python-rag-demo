import pytest
import tempfile
import os
from unittest.mock import Mock, patch
from src.infrastructure.embedding.sentence_transformers_embedding import SentenceTransformersEmbedding
from src.infrastructure.vectorstore.chromadb_store import ChromaDBStore
from src.infrastructure.llm.ollama_llm import OllamaLLM
from src.domain.entities.chunk import Chunk
from src.infrastructure.loader.text_file_loader import TextFileLoader


def test_embedding_service_embed_text():
    service = SentenceTransformersEmbedding()
    embedding = service.embed_text("Hello world")
    assert isinstance(embedding, list)
    assert len(embedding) > 0
    assert all(isinstance(x, float) for x in embedding)


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


def test_ollama_llm_generate():
    with patch("src.infrastructure.llm.ollama_llm.ollama") as mock_ollama:
        mock_ollama.chat.return_value = {"message": {"content": "Test response"}}
        llm = OllamaLLM()
        result = llm.generate("Question?", "Context")
        assert result == "Test response"
        mock_ollama.chat.assert_called_once()


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
