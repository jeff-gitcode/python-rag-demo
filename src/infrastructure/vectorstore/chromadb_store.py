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
