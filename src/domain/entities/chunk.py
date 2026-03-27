from dataclasses import dataclass
from typing import Optional


@dataclass
class Chunk:
    id: str
    document_id: str
    content: str
    embedding: Optional[list[float]] = None
