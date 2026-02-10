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


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and ingest a PDF document into the RAG system.

    This endpoint performs the full ingestion pipeline:
    1. Validates the uploaded file (type and size)
    2. Temporarily stores the file on disk
    3. Extracts and chunks text from the PDF
    4. Generates vector embeddings for each chunk
    5. Persists embeddings and metadata in the vector store
    6. Cleans up temporary files

    Args:
        file (UploadFile): PDF file uploaded by the client.

    Returns:
        DocumentUploadResponse: Metadata about the processed document,
        including the document ID and number of chunks created.

    Raises:
        HTTPException:
            - 400 if the file type or size is invalid
            - 500 if an unexpected error occurs during processing
    """

    # Validate file type early to avoid unnecessary work
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Generate a unique temporary file path
    temp_path = Path(settings.upload_dir) / f"{int(time.time())}_{file.filename}"

    try:
        # Persist uploaded file to disk so downstream services
        # can operate on a file path rather than a stream.
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Enforce maximum file size constraint
        file_size = temp_path.stat().st_size
        if file_size > settings.max_file_size:
            temp_path.unlink()
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Max size: {settings.max_file_size / 1024 / 1024}MB"
            )

        # Extract text and chunk the document
        processing_result = doc_processor.process_document(
            file_path=str(temp_path),
            filename=file.filename
        )

        # Store chunks and embeddings in the vector database
        chunks_added = vector_store.add_documents(
            doc_id=processing_result["doc_id"],
            chunks=processing_result["chunks"],
            filename=processing_result["filename"]
        )

        # Remove temporary file after successful ingestion
        temp_path.unlink()

        return DocumentUploadResponse(
            doc_id=processing_result["doc_id"],
            filename=processing_result["filename"],
            chunks_created=chunks_added,
            message=(
                f"Successfully processed {processing_result['filename']} "
                f"into {chunks_added} chunks"
            )
        )

    except ValueError as exc:
        if temp_path.exists():
            temp_path.unlink()
        raise HTTPException(status_code=400, detail=str(exc))

    except Exception as exc:
        if temp_path.exists():
            temp_path.unlink()
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(exc)}"
        )


@router.delete("/{doc_id}", response_model=DeleteResponse)
async def delete_document(doc_id: str):
    """
    Delete a specific document and all its associated chunks.

    This operation removes all vector embeddings and metadata associated
    with the given document ID from the vector store.

    Args:
        doc_id (str): Unique identifier of the document to delete.

    Returns:
        DeleteResponse: Number of chunks deleted and a status message.

    Raises:
        HTTPException: 404 if the document does not exist.
    """

    deleted_count = vector_store.delete_document(doc_id)

    if deleted_count == 0:
        raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")

    return DeleteResponse(
        deleted_count=deleted_count,
        message=f"Deleted document {doc_id} ({deleted_count} chunks)"
    )


@router.delete("/", response_model=DeleteResponse)
async def delete_all_documents():
    """
    Delete all documents from the vector store.

    This is a destructive operation that removes every stored embedding
    and resets the collection. Intended primarily for development,
    testing, or administrative use.

    Returns:
        DeleteResponse: Total number of chunks deleted.
    """

    deleted_count = vector_store.delete_all()

    return DeleteResponse(
        deleted_count=deleted_count,
        message=f"Deleted all documents ({deleted_count} chunks)"
    )

