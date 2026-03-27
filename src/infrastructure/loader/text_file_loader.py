import uuid
from src.domain.repositories.document_loader import DocumentLoader
from src.domain.entities.document import Document


class TextFileLoader(DocumentLoader):
    def load(self, path: str) -> Document:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return Document(
            id=str(uuid.uuid4()),
            content=content,
            source_path=path
        )
