import pytest
from src.infrastructure.embedding.sentence_transformers_embedding import SentenceTransformersEmbedding
from src.infrastructure.vectorstore.chromadb_store import ChromaDBStore
from src.domain.entities.chunk import Chunk


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
