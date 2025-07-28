"""
LangChain-based embedding function using the same Sentence Transformers model
Maintains compatibility with existing embeddings
"""

from typing import List
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)


class LangChainEmbeddingFunction:
    """LangChain wrapper for SentenceTransformer embeddings"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize with the same free embedding model used in v0.2
        """
        print(f"Loading LangChain embedding model: {model_name}")
        self.embeddings = SentenceTransformerEmbeddings(model_name=model_name)
        self._model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        return self.embeddings.embed_query(text)


def get_embedding_function():
    """Return LangChain-compatible embedding function"""
    # Try better models first, fallback to lighter ones
    model_priority = [
        "all-MiniLM-L12-v2",  # Better quality, ~130MB
        "all-MiniLM-L6-v2",  # Good balance, ~90MB
        "paraphrase-MiniLM-L3-v2",  # Fallback, ~60MB
    ]

    for model_name in model_priority:
        try:
            return LangChainEmbeddingFunction(model_name)
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            continue

    # Ultimate fallback
    print("Using fallback embedding model")
    return LangChainEmbeddingFunction("all-MiniLM-L6-v2")
