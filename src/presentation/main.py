import argparse
from src.infrastructure.loader.text_file_loader import TextFileLoader
from src.infrastructure.embedding.sentence_transformers_embedding import SentenceTransformersEmbedding
from src.infrastructure.vectorstore.chromadb_store import ChromaDBStore
from src.infrastructure.llm.ollama_llm import OllamaLLM
from src.application.ingest_document import IngestDocumentUseCase
from src.application.query_documents import QueryDocumentsUseCase


def main():
    parser = argparse.ArgumentParser(description="Simple RAG CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents")
    ingest_parser.add_argument("path", help="Path to document or directory")
    
    query_parser = subparsers.add_parser("query", help="Query documents")
    query_parser.add_argument("question", help="Question to ask")
    query_parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve")
    
    args = parser.parse_args()
    
    loader = TextFileLoader()
    embedding = SentenceTransformersEmbedding()
    vector_store = ChromaDBStore()
    llm = OllamaLLM()
    
    if args.command == "ingest":
        use_case = IngestDocumentUseCase(
            loader=loader,
            embedding_service=embedding,
            repository=vector_store
        )
        use_case.execute(args.path)
        print(f"Successfully ingested: {args.path}")
    
    elif args.command == "query":
        use_case = QueryDocumentsUseCase(
            embedding_service=embedding,
            repository=vector_store,
            llm_service=llm
        )
        answer = use_case.execute(args.question, args.top_k)
        print(f"\nAnswer: {answer.content}\n")
        print(f"Sources: {answer.sources}")


if __name__ == "__main__":
    main()
