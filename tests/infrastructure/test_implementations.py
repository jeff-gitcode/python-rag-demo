import pytest
from src.infrastructure.embedding.sentence_transformers_embedding import SentenceTransformersEmbedding


def test_embedding_service_embed_text():
    service = SentenceTransformersEmbedding()
    embedding = service.embed_text("Hello world")
    assert isinstance(embedding, list)
    assert len(embedding) > 0
    assert all(isinstance(x, float) for x in embedding)
