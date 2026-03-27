import ollama
from src.domain.repositories.llm_service import LLMService


class OllamaLLM(LLMService):
    def __init__(self, model: str = "llama3.2"):
        self.model = model

    def generate(self, prompt: str, context: str) -> str:
        full_prompt = f"""Context: {context}

Question: {prompt}

Answer:"""

        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": full_prompt}]
        )
        return response["message"]["content"]
