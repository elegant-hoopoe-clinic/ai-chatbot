"""
LangChain-based document processor
Handles PDF loading and text splitting using LangChain components
"""

import os
from typing import List
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


class LangChainDocumentProcessor:
    """Process documents using LangChain components"""

    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

    def load_documents(self, directory_path: str) -> List[Document]:
        """Load all PDF documents from directory using LangChain"""
        if not os.path.exists(directory_path):
            raise ValueError(f"Directory {directory_path} does not exist")

        # Use LangChain's PDF directory loader
        loader = PyPDFDirectoryLoader(directory_path)
        documents = loader.load()

        print(f"📄 Loaded {len(documents)} pages from PDFs")
        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks using LangChain"""
        chunks = self.text_splitter.split_documents(documents)

        # Enhance metadata for better retrieval
        for i, chunk in enumerate(chunks):
            chunk.metadata.update(
                {
                    "chunk_index": i,
                    "chunk_length": len(chunk.page_content),
                    "word_count": len(chunk.page_content.split()),
                }
            )

        print(f"📝 Split into {len(chunks)} chunks")
        return chunks

    def process_documents(self, directory_path: str) -> List[Document]:
        """Complete document processing pipeline"""
        documents = self.load_documents(directory_path)
        chunks = self.split_documents(documents)
        return chunks
