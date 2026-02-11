"""
RAG MVP - FastAPI Application

Summary:
    This module initializes and configures the FastAPI application that
    exposes the RAG (Retrieval-Augmented Generation) backend services
    to external clients (e.g., frontend UI, Postman, other services).

    Architectural Role in the Pipeline:
        1. Configuration Layer:
            Loads environment-aware settings that define application behavior.

        2. Routing Layer:
            Registers API routers that expose document ingestion and query
            endpoints.

        3. Middleware Layer:
            Configures CORS to allow cross-origin communication between
            frontend and backend.

        4. Infrastructure Layer:
            Provides health monitoring endpoints and bootstraps the server.

    This file does NOT contain business logic. It wires together the
    application components and exposes them over HTTP.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.models import HealthResponse
from app.routers import documents, query
from app.services.vector_store import VectorStore


# ---------------------------------------------------------------------
# Application Initialization
# ---------------------------------------------------------------------

app = FastAPI(
    title=settings.app_name,
    description="Production-ready RAG API using Groq LLM and ChromaDB",
    version="1.0.0",
    debug=settings.debug
)


# ---------------------------------------------------------------------
# Middleware Configuration
# ---------------------------------------------------------------------
# CORS (Cross-Origin Resource Sharing) allows frontend applications
# running on different domains or ports (e.g., localhost:3000) to
# communicate with this backend API (localhost:8000).
#
# In production, allow_origins should be restricted to trusted domains.
# ---------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------
# Router Registration
# ---------------------------------------------------------------------
# Routers encapsulate endpoint groups and keep application modular.
#
# - documents.router handles ingestion and deletion of documents.
# - query.router handles RAG-based question answering.
#
# This separation maintains clean architecture boundaries.
# ---------------------------------------------------------------------

app.include_router(documents.router)
app.include_router(query.router)


# ---------------------------------------------------------------------
# Root Endpoint
# ---------------------------------------------------------------------

@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint.

    Provides basic API metadata and navigation hints.

    Returns:
        dict: Basic information about the API including links to:
            - OpenAPI documentation
            - Health check endpoint
    """
    return {
        "message": "RAG MVP API",
        "docs": "/docs",
        "health": "/health"
    }


# ---------------------------------------------------------------------
# Health Check Endpoint
# ---------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.

    Validates that the API and vector store are operational.

    This endpoint is typically used by:
        - Load balancers
        - Monitoring tools
        - Deployment infrastructure
        - DevOps pipelines

    Pipeline Context:
        Even though this endpoint does not perform retrieval or generation,
        it verifies connectivity to the vector database, which is a critical
        dependency of the RAG system.

    Returns:
        HealthResponse: Structured response containing:
            - API status
            - Vector DB connectivity status
            - Total indexed documents
            - Total indexed chunks
    """

    # Instantiate vector store to verify connection
    vector_store = VectorStore(
        persist_directory=settings.chroma_persist_dir,
        collection_name=settings.collection_name
    )

    # Retrieve database statistics
    stats = vector_store.get_stats()

    return HealthResponse(
        status="healthy",
        vector_db_status="connected",
        total_documents=stats['total_documents'],
        total_chunks=stats['total_chunks']
    )


# ---------------------------------------------------------------------
# Local Development Entry Point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    """
    Local development server entry point.

    Runs the FastAPI app using Uvicorn.

    Notes:
        - host="0.0.0.0" makes the app accessible externally.
        - port=8000 exposes the API at localhost:8000.
        - reload=True enables auto-reloading during development.
    """

    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )