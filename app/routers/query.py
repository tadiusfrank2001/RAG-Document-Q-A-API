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



# ---------------------------------------------------------------------
# Query endpoint
# ---------------------------------------------------------------------

@router.post("/", response_model=QueryResponse)
async def query_documents(request: QueryRequest) -> QueryResponse:
    """Execute a RAG-based query over uploaded documents.

    This endpoint represents the full Retrieval-Augmented Generation flow:
    1. Accept a user query from the API client.
    2. Embed the query and search the vector database for relevant chunks.
    3. Use retrieved chunks as context for the LLM.
    4. Generate a grounded answer based on retrieved context.
    5. Return the answer along with source citations and timing metadata.

    Args:
        request (QueryRequest): User query payload containing:
            - query: Natural language question.
            - top_k: Number of relevant chunks to retrieve.

    Returns:
        QueryResponse: Structured response containing:
            - query: Original user query.
            - answer: Generated answer from the LLM.
            - sources: Retrieved document chunks used as context.
            - processing_time: Total request latency in seconds.

    Raises:
        HTTPException:
            - 404 if no relevant documents are found.
            - 500 if an unexpected error occurs during processing.
    """

    start_time = time.time()

    try:
        # -------------------------------------------------------------
        # Step 1: Retrieve relevant document chunks
        # -------------------------------------------------------------
        results = vector_store.search(
            query=request.query,
            top_k=request.top_k
        )

        if not results:
            raise HTTPException(
                status_code=404,
                detail=(
                    "No relevant documents found. "
                    "Please upload documents before querying."
                )
            )

        # -------------------------------------------------------------
        # Step 2: Generate answer using LLM + retrieved context
        # -------------------------------------------------------------
        answer = llm_client.generate_rag_response(
            query=request.query,
            context_chunks=results
        )

        # -------------------------------------------------------------
        # Step 3: Format retrieved chunks as response sources
        # -------------------------------------------------------------
        sources = [
            SourceChunk(
                content=(
                    result["content"][:200] + "..."
                    if len(result["content"]) > 200
                    else result["content"]
                ),
                doc_id=result["metadata"]["doc_id"],
                filename=result["metadata"]["filename"],
                similarity_score=round(result["similarity_score"], 4),
            )
            for result in results
        ]

        processing_time = round(time.time() - start_time, 2)

        # -------------------------------------------------------------
        # Step 4: Return structured API response
        # -------------------------------------------------------------
        return QueryResponse(
            query=request.query,
            answer=answer,
            sources=sources,
            processing_time=processing_time
        )

    except HTTPException:
        # Propagate known HTTP errors unchanged
        raise

    except Exception as e:
        # Catch-all for unexpected failures
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )
    

