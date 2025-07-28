"""
Test script for LangChain RAG system
"""

import os
import shutil
from advanced_langchain_rag import AdvancedLangChainRAG


def test_rag_system():
    """Test the complete RAG system functionality"""
    print("🧪 Testing LangChain RAG System")
    print("=" * 40)

    # Test configuration
    test_chroma_path = "test_chroma"

    try:
        # Clean up any existing test database
        if os.path.exists(test_chroma_path):
            shutil.rmtree(test_chroma_path)

        # Initialize RAG system
        print("1️⃣ Initializing RAG system...")
        rag = AdvancedLangChainRAG(
            data_path="data",
            chroma_path=test_chroma_path,
            chunk_size=300,  # Smaller for testing
            chunk_overlap=50,
        )
        print("✅ RAG system initialized")

        # Test database population
        print("\n2️⃣ Testing database population...")
        if not os.path.exists("data"):
            print("⚠️ No data directory found. Creating dummy data...")
            os.makedirs("data", exist_ok=True)
            # Note: In real usage, PDFs would be here
            print("📝 Please add PDF files to the 'data' directory")
            return False

        try:
            rag.populate_database(reset=True)
            print("✅ Database populated successfully")
        except Exception as e:
            print(f"❌ Database population failed: {e}")
            return False

        # Test query functionality
        print("\n3️⃣ Testing query functionality...")
        test_query = "What is the main content of the documents?"

        try:
            result = rag.query(test_query, k=3)

            if result["answer"].startswith("❌"):
                print(f"⚠️ Query returned error: {result['answer']}")
                return False

            print(f"✅ Query successful")
            print(f"📝 Answer preview: {result['answer'][:100]}...")
            print(f"📚 Found {len(result['sources'])} relevant sources")

        except Exception as e:
            print(f"❌ Query failed: {e}")
            return False

        # Test query with scores
        print("\n4️⃣ Testing query with scores...")
        try:
            result_with_scores = rag.query_with_scores(test_query, k=2)

            if result_with_scores["answer"].startswith("❌"):
                print(f"⚠️ Scored query returned error: {result_with_scores['answer']}")
                return False

            print("✅ Scored query successful")

            for source in result_with_scores["sources"]:
                if "relevance_score" in source:
                    print(f"📊 Source relevance: {source['relevance_score']:.3f}")

        except Exception as e:
            print(f"❌ Scored query failed: {e}")
            return False

        print("\n🎉 All tests passed!")
        return True

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

    finally:
        # Clean up test database
        if os.path.exists(test_chroma_path):
            shutil.rmtree(test_chroma_path)
            print("🧹 Test database cleaned up")


def test_components():
    """Test individual components"""
    print("\n🔍 Testing individual components...")

    # Test embeddings
    try:
        from langchain_embeddings import get_embedding_function

        embedding_func = get_embedding_function()
        print("✅ Embedding function loaded")

        # Test embedding
        test_texts = ["This is a test", "Another test sentence"]
        embeddings = embedding_func.embed_documents(test_texts)
        print(f"✅ Generated embeddings: {len(embeddings)} vectors")

    except Exception as e:
        print(f"❌ Embedding test failed: {e}")
        return False

    # Test OpenAI client
    try:
        from langchain_openai_client import get_openai_client

        client = get_openai_client()

        if client.is_available():
            print("✅ OpenAI client is available")
        else:
            print("⚠️ OpenAI client not available (check API key)")

    except Exception as e:
        print(f"❌ OpenAI client test failed: {e}")
        return False

    return True


if __name__ == "__main__":
    print("🚀 Starting LangChain RAG System Tests")
    print("=" * 50)

    # Test components first
    components_ok = test_components()

    if components_ok:
        # Test full system
        system_ok = test_rag_system()

        if system_ok:
            print("\n🎊 All tests completed successfully!")
            print("💡 You can now use the system with:")
            print("   python interactive_rag.py")
        else:
            print("\n❌ System tests failed")
    else:
        print("\n❌ Component tests failed")

    print("\n" + "=" * 50)
