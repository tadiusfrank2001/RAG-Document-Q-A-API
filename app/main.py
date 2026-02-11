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

