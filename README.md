# RAG MVP - Document Q&A API

A production-ready Retrieval-Augmented Generation (RAG) system using Groq LLM and ChromaDB.

## Features

✅ **PDF Document Processing** - Upload PDFs and automatically chunk them  
✅ **Semantic Search** - Vector-based similarity search using embeddings  
✅ **Groq LLM Integration** - Fast, free AI responses using Llama 3.3  
✅ **ChromaDB Vector Store** - Local, persistent vector database  
✅ **REST API** - Clean, documented FastAPI endpoints  
✅ **Docker Support** - Containerized for easy deployment  

## Architecture

```
User → FastAPI → [Upload PDF] → Extract Text → Chunk → Embed → ChromaDB
                ↓
         [Query] → Embed Query → Search ChromaDB → Get Context → Groq LLM → Answer
```

## Quick Start

### 1. Prerequisites

- Python 3.11+
- Groq API key (get free at [console.groq.com](https://console.groq.com/keys))

### 2. Installation

```bash
# Clone the repo
git clone <your-repo>
cd rag-mvp

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### 3. Run Locally

```bash
# Start the API
python -m uvicorn app.main:app --reload

# API runs on http://localhost:8000
# Docs available at http://localhost:8000/docs
```

### 4. Run with Docker

```bash
# Build and run
docker-compose up --build

# API runs on http://localhost:8000
```

## API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

### Upload Document

```bash
curl -X POST "http://localhost:8000/documents/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your-document.pdf"
```

Response:
```json
{
  "doc_id": "doc_abc123def456",
  "filename": "your-document.pdf",
  "chunks_created": 25,
  "status": "success",
  "message": "Successfully processed your-document.pdf into 25 chunks"
}
```

### Query Documents

```bash
curl -X POST "http://localhost:8000/query/" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the main topic of the document?",
    "top_k": 5
  }'
```

Response:
```json
{
  "query": "What is the main topic of the document?",
  "answer": "Based on the provided context, the main topic...",
  "sources": [
    {
      "content": "Excerpt from the document...",
      "doc_id": "doc_abc123def456",
      "filename": "your-document.pdf",
      "similarity_score": 0.8234
    }
  ],
  "processing_time": 1.23
}
```

### Delete Document

```bash
# Delete specific document
curl -X DELETE "http://localhost:8000/documents/doc_abc123def456"

# Delete ALL documents
curl -X DELETE "http://localhost:8000/documents/"
```

## Interactive API Docs

Visit http://localhost:8000/docs for the interactive Swagger UI where you can test all endpoints.

## Project Structure

```
rag-mvp/
├── app/
│   ├── main.py              # FastAPI application
│   ├── config.py            # Settings & configuration
│   ├── models.py            # Pydantic schemas
│   ├── routers/
│   │   ├── documents.py     # Upload/delete endpoints
│   │   └── query.py         # RAG query endpoint
│   └── services/
│       ├── document_processor.py  # PDF processing
│       ├── embeddings.py          # Text → vectors
│       ├── vector_store.py        # ChromaDB wrapper
│       └── llm.py                 # Groq client
├── data/
│   ├── uploads/             # Temporary PDF storage
│   └── chroma_db/           # Vector database (persistent)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env
```

## Configuration

Edit `.env` to customize:

```bash
# Required
GROQ_API_KEY=your_key_here

# Optional
GROQ_MODEL=llama-3.3-70b-versatile  # Groq model to use
EMBEDDING_MODEL=all-MiniLM-L6-v2     # SentenceTransformer model
CHUNK_SIZE=500                       # Text chunk size
CHUNK_OVERLAP=50                     # Overlap between chunks
RETRIEVAL_TOP_K=5                    # Number of chunks to retrieve
```

## Why This Stack?

- **Groq**: Free tier, extremely fast inference (500+ tokens/sec)
- **ChromaDB**: Lightweight, persistent, no external dependencies
- **SentenceTransformers**: Open-source, runs locally, no API costs
- **FastAPI**: Auto-documentation, async support, type safety

## Limitations

- **PDF only**: Currently only supports PDF files
- **Local storage**: ChromaDB runs locally (great for MVP, scales to millions of vectors)
- **No authentication**: Add your own auth layer for production
- **No rate limiting**: Add middleware if needed

## Next Steps

After validating the MVP:
- Add rate limiting
- Add authentication (API keys, OAuth)
- Deploy to AWS/Vercel/Railway
- Add support for more file types (DOCX, TXT, etc.)
- Implement caching for common queries
- Add monitoring and logging

## Troubleshooting

### "No module named 'app'"
Make sure you're running from the project root:
```bash
cd rag-mvp
python -m uvicorn app.main:app --reload
```

### "Failed to extract text from PDF"
PDF might be image-based. Consider adding OCR support with `pytesseract`.

### "Groq API error"
Check your API key in `.env` and ensure you have credits/quota.

## License

MIT

## Support

For issues, open a GitHub issue or contact the maintainer.