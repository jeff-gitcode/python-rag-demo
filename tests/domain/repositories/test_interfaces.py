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
