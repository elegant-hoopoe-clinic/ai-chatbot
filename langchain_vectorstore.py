"""
LangChain-based vector store using ChromaDB
Maintains compatibility with existing data while using LangChain components
"""

import os
import shutil
from typing import List, Optional
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_embeddings import get_embedding_function


class LangChainVectorStore:
    """Vector store using LangChain's Chroma integration"""

    def __init__(
        self, persist_directory: str = "chroma", collection_name: str = "documents"
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_function = get_embedding_function()
        self._vectorstore = None

    def clear_database(self):
        """Clear the existing database"""
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
            print("✅ Database cleared")

    def create_vectorstore(self, documents: List[Document]) -> Chroma:
        """Create a new vector store from documents"""
        if not documents:
            raise ValueError("No documents provided")

        # Create the vector store
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_function,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
        )

        print(f"✅ Created vector store with {len(documents)} documents")
        self._vectorstore = vectorstore
        return vectorstore

    def get_vectorstore(self) -> Optional[Chroma]:
        """Get existing vector store"""
        if not os.path.exists(self.persist_directory):
            print("❌ Database not found. Run populate script first")
            return None

        try:
            vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_function,
                collection_name=self.collection_name,
            )
            self._vectorstore = vectorstore
            return vectorstore

        except Exception as e:
            print(f"❌ Error loading database: {e}")
            return None

    def add_documents(self, documents: List[Document]):
        """Add documents to existing vector store"""
        if self._vectorstore is None:
            self._vectorstore = self.get_vectorstore()

        if self._vectorstore is None:
            raise ValueError("No vector store available. Create one first.")

        self._vectorstore.add_documents(documents)
        print(f"✅ Added {len(documents)} documents to vector store")

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Search for similar documents"""
        if self._vectorstore is None:
            self._vectorstore = self.get_vectorstore()

        if self._vectorstore is None:
            return []

        results = self._vectorstore.similarity_search(query, k=k)
        return results

    def similarity_search_with_score(self, query: str, k: int = 5) -> List[tuple]:
        """Search for similar documents with relevance scores"""
        if self._vectorstore is None:
            self._vectorstore = self.get_vectorstore()

        if self._vectorstore is None:
            return []

        results = self._vectorstore.similarity_search_with_score(query, k=k)
        return results
