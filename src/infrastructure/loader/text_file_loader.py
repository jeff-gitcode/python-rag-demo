import os
import uuid
from src.domain.repositories.document_loader import DocumentLoader
from src.domain.entities.document import Document


class TextFileLoader(DocumentLoader):
    def load(self, path: str) -> Document:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        if not os.path.isfile(path):
            raise ValueError(f"Path is not a file: {path}")
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
        except IOError as e:
            raise IOError(f"Error reading file {path}: {e}")
        
        return Document(
            id=str(uuid.uuid4()),
            content=content,
            source_path=path
        )
