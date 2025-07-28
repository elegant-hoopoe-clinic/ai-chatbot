"""
Interactive CLI for the LangChain RAG system
"""

import os
from advanced_langchain_rag import AdvancedLangChainRAG


def main():
    print("🤖 LangChain RAG System - Interactive Mode")
    print("=" * 50)

    # Initialize RAG system
    rag_system = AdvancedLangChainRAG()

    # Check if database exists
    if not os.path.exists("chroma"):
        print("❌ Database not found!")
        print("Please run: python populate_database.py")
        return

    print("✅ RAG system initialized")
    print("💡 Type 'quit' or 'exit' to stop")
    print("💡 Type 'reset' to clear and repopulate database")
    print("=" * 50)

    while True:
        try:
            # Get user input
            query = input("\n🔍 Ask a question: ").strip()

            if not query:
                continue

            if query.lower() in ["quit", "exit", "q"]:
                print("👋 Goodbye!")
                break

            if query.lower() == "reset":
                print("🔄 Resetting database...")
                rag_system.populate_database(reset=True)
                continue

            # Process query
            print("\n🤔 Thinking...")
            result = rag_system.query(query)

            # Display results
            print("\n" + "=" * 50)
            print("🎯 ANSWER:")
            print("=" * 50)
            print(result["answer"])

            if result.get("sources"):
                print("\n" + "=" * 30)
                print("📚 SOURCES:")
                print("=" * 30)

                for source in result["sources"]:
                    print(f"\n{source['index']}. {source['content']}")
                    if source.get("metadata", {}).get("source"):
                        source_file = os.path.basename(source["metadata"]["source"])
                        print(f"   📄 {source_file}")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
