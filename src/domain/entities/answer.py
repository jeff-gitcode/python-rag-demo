from dataclasses import dataclass
from typing import Optional


@dataclass
class Answer:
    id: str
    query_id: str
    content: str
    sources: list[str]
