"""
RAG Query API Endpoints.

This module defines the FastAPI routing layer for querying documents using
a Retrieval-Augmented Generation (RAG) pipeline.

At this stage of the system, all core backend services have already been
implemented:
- Configuration management via Pydantic settings
- Strongly-typed API schemas using Pydantic models
- Document ingestion and chunking
- Embedding generation for documents and queries
- Vector database storage and retrieval (ChromaDB)
- Large Language Model (LLM) integration for answer generation

This router acts as the boundary between the external world (UI / client)
and the internal RAG pipeline. It accepts user queries, orchestrates
retrieval from the vector store, injects retrieved context into the LLM,
and returns a structured, traceable response to the client.
"""


from fastapi import APIRouter, HTTPException
import time

from app.models import QueryRequest, QueryResponse, SourceChunk
from app.config import settings
from app.services.vector_store import VectorStore
from app.services.llm import LLMClient



# ---------------------------------------------------------------------
# Router configuration
# ---------------------------------------------------------------------

router = APIRouter(prefix="/query", tags=["Query"])


# ---------------------------------------------------------------------
# Service initialization
# ---------------------------------------------------------------------
# These services encapsulate core RAG logic and are reused across requests.
# In larger systems, these would typically be injected via FastAPI
# dependencies rather than instantiated at import time.

vector_store = VectorStore(
    persist_directory=settings.chroma_persist_dir,
    collection_name=settings.collection_name
)

llm_client = LLMClient(
    api_key=settings.groq_api_key,
    model=settings.groq_model
)




