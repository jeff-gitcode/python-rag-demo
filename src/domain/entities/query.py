from dataclasses import dataclass
from typing import Optional


@dataclass
class Query:
    id: str
    text: str
    embedding: Optional[list[float]] = None
