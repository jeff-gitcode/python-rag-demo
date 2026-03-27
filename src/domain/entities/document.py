from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Document:
    id: str
    content: str
    source_path: str
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
