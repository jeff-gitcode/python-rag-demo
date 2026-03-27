from abc import ABC, abstractmethod
from src.domain.entities.document import Document


class DocumentLoader(ABC):
    @abstractmethod
    def load(self, path: str) -> Document:
        pass
