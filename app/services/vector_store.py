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



    def add_documents(
        self,
        doc_id: str,
        chunks: List[str],
        filename: str,
    ) -> int:
        """Add embedded document chunks to the vector store.

        Each chunk is embedded, assigned a unique identifier, and stored
        along with metadata to support retrieval, filtering, and deletion.

        Args:
            doc_id (str): Stable document identifier.
            chunks (List[str]): Text chunks produced by the document processor.
            filename (str): Original filename for metadata tracking.

        Returns:
            int: Number of chunks successfully added.
        """
        print(f"Generating embeddings for {len(chunks)} chunks...")
        embeddings = self.embedding_service.embed_batch(chunks)

        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]

        metadatas = [
            {
                "doc_id": doc_id,
                "filename": filename,
                "chunk_index": i,
            }
            for i in range(len(chunks))
        ]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )

        print(f"Added {len(chunks)} chunks to vector store")
        return len(chunks)




    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Perform a similarity search over stored document chunks.

        The query is embedded using the same embedding model as the
        documents, enabling cosine similarity comparison in vector space.

        Args:
            query (str): User query string.
            top_k (int): Number of most similar chunks to retrieve.

        Returns:
            List[Dict]: List of matched document chunks containing:
                - content (str): Chunk text
                - metadata (Dict): Associated metadata
                - distance (float): Vector distance
                - similarity_score (float): Normalized similarity score
        """
        query_embedding = self.embedding_service.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        formatted_results = []
        if results["ids"]:
            for i in range(len(results["ids"][0])):
                distance = results["distances"][0][i]
                formatted_results.append(
                    {
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": distance,
                        "similarity_score": 1 - distance,
                    }
                )

        return formatted_results




    def delete_document(self, doc_id: str) -> int:
        """Delete all stored chunks associated with a document.

        This operation removes every vector belonging to a document,
        enabling safe re-ingestion or permanent deletion.

        Args:
            doc_id (str): Document identifier.

        Returns:
            int: Number of chunks deleted.
        """
        results = self.collection.get(
            where={"doc_id": doc_id},
            include=["metadatas"],
        )

        if not results["ids"]:
            return 0

        self.collection.delete(ids=results["ids"])

        deleted_count = len(results["ids"])
        print(f"Deleted {deleted_count} chunks for doc: {doc_id}")
        return deleted_count
    



    def delete_all(self) -> int:
        """Delete all vectors from the collection.

        This resets the vector store by deleting and recreating the
        collection. Intended for development, testing, or full reindexing.

        Returns:
            int: Number of chunks deleted.
        """
        count = self.collection.count()

        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        print(f"Deleted all {count} chunks from vector store")
        return count
    


    def get_stats(self) -> Dict:
        """Retrieve high-level statistics about the vector store.

        Returns:
            Dict: Vector store statistics including:
                - total_documents (int)
                - total_chunks (int)
                - collection_name (str)
        """
        total_chunks = self.collection.count()

        if total_chunks > 0:
            all_metadata = self.collection.get(include=["metadatas"])
            unique_docs = {m["doc_id"] for m in all_metadata["metadatas"]}
            num_docs = len(unique_docs)
        else:
            num_docs = 0

        return {
            "total_documents": num_docs,
            "total_chunks": total_chunks,
            "collection_name": self.collection_name,
        }
    

