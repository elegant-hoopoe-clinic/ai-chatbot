"""
LangChain RAG database population script
"""

import argparse
from advanced_langchain_rag import AdvancedLangChainRAG


def main():
    parser = argparse.ArgumentParser(
        description="Populate RAG database using LangChain"
    )
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    parser.add_argument("--data-path", default="data", help="Path to data directory")
    parser.add_argument(
        "--chroma-path", default="chroma", help="Path to ChromaDB directory"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=600, help="Chunk size for text splitting"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
        help="Chunk overlap for text splitting",
    )

    args = parser.parse_args()

    # Initialize RAG system
    rag_system = AdvancedLangChainRAG(
        data_path=args.data_path,
        chroma_path=args.chroma_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    # Populate database
    try:
        rag_system.populate_database(reset=args.reset)
        print("🎉 Database population completed successfully!")

    except Exception as e:
        print(f"❌ Error during database population: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
