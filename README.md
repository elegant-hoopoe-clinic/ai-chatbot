# LangChain RAG System v0.3

A modern Retrieval-Augmented Generation (RAG) system built with LangChain, featuring OpenAI integration and the same proven embedding models from previous versions.

## 🎯 Key Features

- **LangChain Integration**: Built using LangChain components for better modularity and extensibility
- **OpenAI Only**: Uses only OpenAI API for LLM queries (no local LLMs)
- **Same Embedding Model**: Maintains compatibility with existing embeddings using Sentence Transformers
- **ChromaDB Vector Store**: Efficient vector storage and retrieval
- **Advanced Text Processing**: Improved document chunking and metadata handling
- **Interactive CLI**: User-friendly command-line interface
- **Comprehensive APIs**: Both simple and advanced query methods

## 🏗️ Architecture

```
📁 v0.3/
├── 📄 advanced_langchain_rag.py      # Main RAG system class
├── 📄 langchain_embeddings.py        # LangChain embedding wrapper
├── 📄 langchain_document_processor.py # Document loading and splitting
├── 📄 langchain_vectorstore.py       # ChromaDB vector store wrapper
├── 📄 langchain_openai_client.py     # OpenAI client using LangChain
├── 📄 populate_database.py           # Database population script
├── 📄 query_database.py              # Query script with options
├── 📄 interactive_rag.py             # Interactive CLI interface
└── 📄 requirements.txt               # Dependencies
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set OpenAI API Key

```bash
# Windows
set OPENAI_API_KEY=your_api_key_here

# Linux/Mac
export OPENAI_API_KEY=your_api_key_here
```

### 3. Add Your Documents

Place your PDF files in the `data/` directory:

```
data/
├── document1.pdf
├── document2.pdf
└── document3.pdf
```

### 4. Populate Database

```bash
python populate_database.py --reset
```

### 5. Query the System

**Interactive Mode:**
```bash
python interactive_rag.py
```

**Command Line:**
```bash
python query_database.py "What is the main topic of the documents?"
```

## 📖 Usage Examples

### Basic Query
```bash
python query_database.py "Explain the key concepts"
```

### Advanced Query with Scores
```bash
python query_database.py "What are the main findings?" --with-scores --k 3
```

### JSON Output
```bash
python query_database.py "Summarize the content" --json-output
```

### Custom Model
```bash
python query_database.py "What is this about?" --model gpt-4
```

## 🔧 Configuration Options

### Database Population
```bash
python populate_database.py [OPTIONS]

Options:
  --reset              Reset the database before population
  --data-path PATH     Path to data directory (default: data)
  --chroma-path PATH   Path to ChromaDB directory (default: chroma)
  --chunk-size SIZE    Chunk size for text splitting (default: 600)
  --chunk-overlap SIZE Chunk overlap (default: 100)
```

### Querying
```bash
python query_database.py QUERY [OPTIONS]

Options:
  --model MODEL           OpenAI model (default: gpt-3.5-turbo)
  --k NUMBER             Number of documents to retrieve (default: 5)
  --with-scores          Include relevance scores
  --score-threshold NUM  Minimum relevance score (default: 0.0)
  --json-output          Output as JSON
```

## 🧩 Python API Usage

```python
from advanced_langchain_rag import AdvancedLangChainRAG

# Initialize the system
rag = AdvancedLangChainRAG(
    data_path="data",
    chroma_path="chroma",
    chunk_size=600,
    chunk_overlap=100
)

# Populate database
rag.populate_database(reset=True)

# Simple query
result = rag.query("What are the main topics?")
print(result["answer"])

# Query with scores
result = rag.query_with_scores(
    "Explain the methodology", 
    k=3, 
    score_threshold=0.5
)
print(f"Answer: {result['answer']}")
for source in result["sources"]:
    print(f"Score: {source['relevance_score']:.3f}")
```

## 🔄 Migration from v0.2

The v0.3 system maintains compatibility with existing data while providing enhanced functionality:

### Key Changes:
- **LangChain Components**: All core functionality now uses LangChain
- **Same Embeddings**: Uses identical Sentence Transformer models
- **OpenAI Only**: Removed local LLM support (Ollama)
- **Enhanced Metadata**: Better document tracking and source attribution
- **Improved API**: More consistent and feature-rich interfaces

### Migration Steps:
1. Install new requirements
2. Copy your `data/` directory to v0.3
3. Repopulate the database: `python populate_database.py --reset`
4. Test with: `python interactive_rag.py`

## 🛠️ Advanced Features

### Custom Embedding Models
```python
# Modify langchain_embeddings.py to use different models
model_priority = [
    "all-MiniLM-L12-v2",    # Better quality
    "all-mpnet-base-v2",     # Highest quality
    "all-MiniLM-L6-v2",     # Good balance
]
```

### Custom Prompt Templates
```python
rag.prompt_template = """
Based on the provided context, answer the question with specific examples.

Context: {context}
Question: {question}
Answer with examples:
"""
```

### Batch Processing
```python
queries = ["Question 1", "Question 2", "Question 3"]
results = [rag.query(q) for q in queries]
```

## 🐛 Troubleshooting

### Common Issues:

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **OpenAI API Issues**: Check your API key and quota
   ```bash
   python -c "from langchain_openai_client import get_openai_client; print(get_openai_client().is_available())"
   ```

3. **Database Not Found**: Populate the database first
   ```bash
   python populate_database.py --reset
   ```

4. **Memory Issues**: Reduce chunk size or batch size
   ```bash
   python populate_database.py --chunk-size 400
   ```

## 📊 Performance Notes

- **Embedding Model**: Uses the same efficient Sentence Transformers models
- **Chunk Size**: Optimized for 600 characters with 100 character overlap
- **Vector Search**: ChromaDB provides fast similarity search
- **Memory Usage**: Efficient document processing with streaming

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

This project is licensed under the MIT License.
