"""

Embedding generation for the RAG pipeline using sentence-transformers.

This module implements the embedding stage of the RAG pipeline, where
semantically meaningful text chunks are transformed into high-dimensional
vector representations. These embeddings enable similarity-based retrieval
by allowing both documents and queries to be compared in a shared vector
space.

At this stage of the pipeline:
- Text chunks produced by the document processor are embedded into vectors
- Query strings are embedded using the same model to ensure vector
  compatibility
- Output embeddings are passed downstream to the vector store for indexing
  and retrieval

Using a single, consistent embedding model ensures that semantic similarity
between queries and document chunks can be accurately measured.

"""

from sentence_transformers import SentenceTransformer
from typing import List

class EmbeddingService:
    """Generates vector embeddings for text inputs.

    This class encapsulates the logic for converting text into numerical
    embeddings using a sentence-transformer model. It provides a unified
    interface for embedding document chunks and search queries, ensuring
    consistency across the RAG pipeline.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the embedding service.

        Loads the sentence-transformer model and records the dimensionality
        of the resulting embeddings.

        Args:
            model_name (str): Name of the sentence-transformer model to use.
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {self.dimension}")



    def embed_text(self, text: str) -> List[float]:
        """Generate an embedding for a single text input.

        This method is typically used for embedding individual document
        chunks prior to storage in the vector database.

        Args:
            text (str): Text to embed.

        Returns:
            List[float]: High-dimensional embedding vector representing
            the semantic meaning of the input text.
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    


    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of text inputs.

        Batch embedding improves throughput when processing multiple
        document chunks during ingestion.

        Args:
            texts (List[str]): List of text strings to embed.

        Returns:
            List[List[float]]: List of embedding vectors corresponding
            to each input text.
        """
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        return embeddings.tolist()

    
    def embed_query(self, query: str) -> List[float]:
        """Generate an embedding for a search query.

        Queries must be embedded using the same model as document chunks
        to ensure valid similarity comparisons during retrieval.

        Args:
            query (str): User search query.

        Returns:
            List[float]: Embedding vector for the query.
        """
        return self.embed_text(query)
    