"""
Document management endpoints for the RAG pipeline.

This module defines FastAPI routes responsible for ingesting, storing,
and deleting documents used by the Retrieval-Augmented Generation (RAG)
system.

At this stage of the pipeline, raw user inputs (PDF files) are validated
and handed off to lower-level services that:
- extract and chunk text
- generate vector embeddings
- persist embeddings and metadata in the vector database

This layer acts as an orchestration boundary between HTTP requests and
the internal service abstractions, ensuring clean separation between
API concerns and core business logic.
"""


from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import shutil
import time

from app.models import DocumentUploadResponse, DeleteResponse
from app.config import settings
from app.services.document_processor import DocumentProcessor
from app.services.vector_store import VectorStore


router = APIRouter(prefix="/documents", tags=["Documents"])

# ---------------------------------------------------------------------
# Service initialization
# ---------------------------------------------------------------------
# These services encapsulate core pipeline logic and are reused by
# multiple endpoints. In a larger system, these would typically be
# injected via FastAPI dependencies.
doc_processor = DocumentProcessor(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap
)

vector_store = VectorStore(
    persist_directory=settings.chroma_persist_dir,
    collection_name=settings.collection_name
)

