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
