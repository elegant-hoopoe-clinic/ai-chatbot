"""
LangChain RAG query script
"""

import argparse
import json
from advanced_langchain_rag import AdvancedLangChainRAG


def main():
    parser = argparse.ArgumentParser(description="Query RAG system using LangChain")
    parser.add_argument("query_text", type=str, help="The query text.")
    parser.add_argument(
        "--model", type=str, default="gpt-4.1-nano", help="OpenAI model to use"
    )
    parser.add_argument(
        "--k", type=int, default=5, help="Number of documents to retrieve"
    )
    parser.add_argument(
        "--chroma-path", default="chroma", help="Path to ChromaDB directory"
    )
    parser.add_argument(
        "--with-scores", action="store_true", help="Include relevance scores"
    )
    parser.add_argument(
        "--score-threshold", type=float, default=0.0, help="Minimum relevance score"
    )
    parser.add_argument(
        "--json-output", action="store_true", help="Output results as JSON"
    )

    args = parser.parse_args()

    # Initialize RAG system
    rag_system = AdvancedLangChainRAG(chroma_path=args.chroma_path)

    # Query the system
    try:
        if args.with_scores:
            result = rag_system.query_with_scores(
                query_text=args.query_text,
                model=args.model,
                k=args.k,
                score_threshold=args.score_threshold,
            )
        else:
            result = rag_system.query(
                query_text=args.query_text, model=args.model, k=args.k
            )

        # Output results
        if args.json_output:
            print(json.dumps(result, indent=2))
        else:
            display_results(result, args.with_scores)

    except Exception as e:
        print(f"❌ Error during query: {e}")
        return 1

    return 0


def display_results(result: dict, with_scores: bool = False):
    """Display query results in a formatted way"""
    print("=" * 50)
    print("ANSWER:")
    print("=" * 50)
    print(result["answer"])

    if result.get("sources"):
        print("\n" + "=" * 50)
        print("SOURCES:")
        print("=" * 50)

        for source in result["sources"]:
            print(f"\n{source['index']}. {source['content']}")

            if with_scores and "relevance_score" in source:
                print(f"   Relevance Score: {source['relevance_score']:.4f}")

            if source.get("metadata"):
                metadata = source["metadata"]
                if "source" in metadata:
                    print(f"   Source: {metadata['source']}")
                if "page" in metadata:
                    print(f"   Page: {metadata['page']}")


if __name__ == "__main__":
    exit(main())
