"""
API Schemas for the RAG MVP using Pydantic.

This module defines all request and response models for the RAG API, 
including document uploads, queries, deletion, and health checks. 
Models enforce type validation, constraints, and proper data shapes 
before data enters the RAG pipeline, ensuring safe and consistent 
interactions between the API, vector store, and LLM components.

Each model uses Google-style docstrings to describe attributes, 
types, and constraints, which also integrates with FastAPI 
automatic documentation (Swagger/OpenAPI).

"""
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class DocumentUploadResponse(BaseModel):
    """Response returned after a document upload.

    Attributes:
        doc_id (str): Unique identifier of the uploaded document.
        filename (str): Name of the uploaded file.
        chunks_created (int): Number of document chunks created for retrieval.
        status (str): Status of the upload (default is 'success').
        message (str): Informational message regarding the upload.
    """
    doc_id: str
    filename: str
    chunks_created: int
    status: str = "success"
    message: str


class QueryRequest(BaseModel):
    """Schema for a RAG query request.

    Attributes:
        query (str): Natural language question from the user.
            Must be between 1 and 500 characters.
        top_k (int): Number of top similar document chunks to retrieve.
            Must be between 1 and 20.
    """
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)


class SourceChunk(BaseModel):
    """A document chunk retrieved from the vector store.

    Attributes:
        content (str): Text content of the chunk.
        doc_id (str): Unique identifier of the source document.
        filename (str): Original filename of the document.
        similarity_score (float): Similarity score between query and chunk.
    """
    content: str
    doc_id: str
    filename: str
    similarity_score: float


class QueryResponse(BaseModel):
    """Response returned after processing a RAG query.

    Attributes:
        query (str): Original user query.
        answer (str): Generated answer from the LLM.
        sources (List[SourceChunk]): Retrieved document chunks used as context.
        processing_time (float): Total processing time in seconds.
    """

    query: str
    answer: str
    sources: List[SourceChunk]
    processing_time: float


class DeleteResponse(BaseModel):
    """Response returned after deleting documents.

    Attributes:
        deleted_count (int): Number of documents successfully deleted.
        status (str): Status of the delete operation (default is 'success').
        message (str): Informational message regarding the delete operation.
    """

    deleted_count: int
    status: str = "success"
    message: str


class HealthResponse(BaseModel):
    """Schema for a health check response.

    Attributes:
        status (str): Overall status of the API.
        vector_db_status (str): Status of the vector store.
        total_documents (int): Total number of documents currently stored.
        total_chunks (int): Total number of document chunks currently stored.
    """
    
    status: str
    vector_db_status: str
    total_documents: int
    total_chunks: int