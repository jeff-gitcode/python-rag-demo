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
