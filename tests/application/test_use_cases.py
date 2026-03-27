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
