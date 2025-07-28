"""
Advanced LangChain RAG system
Combines all components into a complete RAG pipeline
"""

from typing import List, Optional
from langchain.schema import Document
from langchain_document_processor import LangChainDocumentProcessor
from langchain_vectorstore import LangChainVectorStore
from langchain_openai_client import get_openai_client


class AdvancedLangChainRAG:
    """Complete RAG system using LangChain components"""

    def __init__(
        self,
        data_path: str = "data",
        chroma_path: str = "chroma",
        chunk_size: int = 600,
        chunk_overlap: int = 100,
    ):

        self.data_path = data_path
        self.chroma_path = chroma_path

        # Initialize components
        self.document_processor = LangChainDocumentProcessor(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.vectorstore = LangChainVectorStore(
            persist_directory=chroma_path, collection_name="documents"
        )

        # Prompt template
        self.prompt_template = """Answer the question based on the context below.

Context:
{context}

Question: {question}

Answer:"""

    def populate_database(self, reset: bool = False):
        """Populate the vector database with documents"""
        if reset:
            print("✨ Clearing Database")
            self.vectorstore.clear_database()

        # Process documents
        documents = self.document_processor.process_documents(self.data_path)

        if not documents:
            print("❌ No documents found to process")
            return

        # Create vector store
        self.vectorstore.create_vectorstore(documents)
        print(f"✅ Database populated with {len(documents)} document chunks")

    def query(self, query_text: str, model: str = "gpt-4.1-nano", k: int = 5) -> dict:
        """Query the RAG system"""

        # Search for relevant documents
        relevant_docs = self.vectorstore.similarity_search(query_text, k=k)

        if not relevant_docs:
            return {"answer": "❌ No relevant documents found", "sources": []}

        # Format context
        context_text = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])

        # Create prompt
        prompt = self.prompt_template.format(context=context_text, question=query_text)

        # Generate response using OpenAI
        client = get_openai_client(model)

        if not client.is_available():
            return {"answer": "❌ OpenAI API not available", "sources": []}

        response = client.generate(prompt)

        # Format sources
        sources = []
        for i, doc in enumerate(relevant_docs, 1):
            source_info = {
                "index": i,
                "content": (
                    doc.page_content[:200] + "..."
                    if len(doc.page_content) > 200
                    else doc.page_content
                ),
                "metadata": doc.metadata,
            }
            sources.append(source_info)

        return {"answer": response, "sources": sources, "context": context_text}

    def query_with_scores(
        self,
        query_text: str,
        model: str = "gpt-4.1-nano",
        k: int = 5,
        score_threshold: float = 0.0,
    ) -> dict:
        """Query with relevance scores"""

        # Search with scores
        results = self.vectorstore.similarity_search_with_score(query_text, k=k)

        if not results:
            return {"answer": "❌ No relevant documents found", "sources": []}

        # Filter by score threshold
        filtered_results = [
            (doc, score) for doc, score in results if score >= score_threshold
        ]

        if not filtered_results:
            return {
                "answer": "❌ No documents meet the relevance threshold",
                "sources": [],
            }

        # Extract documents
        relevant_docs = [doc for doc, score in filtered_results]

        # Format context
        context_text = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])

        # Create prompt
        prompt = self.prompt_template.format(context=context_text, question=query_text)

        # Generate response using OpenAI
        client = get_openai_client(model)

        if not client.is_available():
            return {"answer": "❌ OpenAI API not available", "sources": []}

        response = client.generate(prompt)

        # Format sources with scores
        sources = []
        for i, (doc, score) in enumerate(filtered_results, 1):
            source_info = {
                "index": i,
                "content": (
                    doc.page_content[:200] + "..."
                    if len(doc.page_content) > 200
                    else doc.page_content
                ),
                "metadata": doc.metadata,
                "relevance_score": float(score),
            }
            sources.append(source_info)

        return {"answer": response, "sources": sources, "context": context_text}
