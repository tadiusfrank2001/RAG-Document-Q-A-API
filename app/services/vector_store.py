"""
Vector storage and retrieval layer for the RAG pipeline using ChromaDB.

This module implements the vector database layer of the RAG pipeline.
It is responsible for persisting, querying, and managing high-dimensional
vector embeddings produced from document chunks and user queries.

At this stage of the pipeline:
- Embedded document chunks are stored as vectors in a persistent database
- User queries are embedded and compared against stored vectors
- Similarity search retrieves the most relevant document chunks
- CRUD operations enable lifecycle management of indexed documents

This layer acts as the semantic memory of the system, enabling efficient
similarity-based retrieval that powers the generation stage of RAG.

"""


import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Optional
from app.services.embeddings import EmbeddingService


class VectorStore:
    """ChromaDB-backed vector store for document chunk embeddings.

    This class encapsulates all interactions with the vector database,
    including indexing document chunks, performing similarity searches,
    and managing stored vectors across the document lifecycle.
    """

    def __init__(
        self,
        persist_directory: str,
        collection_name: str = "documents",
        embedding_service: Optional[EmbeddingService] = None,
    ):
        """Initialize the vector store and ChromaDB collection.

        This sets up a persistent ChromaDB client, initializes or loads
        a collection, and prepares an embedding service for indexing
        and querying operations.

        Args:
            persist_directory (str): Filesystem path for persistent
                vector storage.
            collection_name (str): Name of the ChromaDB collection.
            embedding_service (EmbeddingService, optional): Service used
                to generate embeddings. If not provided, a default
                embedding service is created.
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        self.embedding_service = embedding_service or EmbeddingService()

        print(f"Vector store initialized. Collection: {collection_name}")
        print(f"Current document count: {self.collection.count()}")
