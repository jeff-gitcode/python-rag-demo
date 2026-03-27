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
