from abc import ABC, abstractmethod
from typing import Optional


class LLMService(ABC):
    @abstractmethod
    def generate(self, prompt: str, context: str) -> str:
        pass
