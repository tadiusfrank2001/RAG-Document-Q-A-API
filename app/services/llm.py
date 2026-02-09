"""
LLM client and agent logic for Retrieval-Augmented Generation (RAG).

This module represents the final stage of the RAG pipeline, where retrieved
context from the vector store is combined with a user query and passed to a
large language model (LLM) to generate a grounded response.

Pipeline context:
1. Configuration defines environment-aware behavior and raw data contracts.
2. Pydantic schemas enforce data shape between API boundaries.
3. Documents are processed into text chunks with stable document identifiers.
4. Embeddings are generated for both document chunks and user queries.
5. A vector store persists embeddings and supports CRUD + similarity search.
6. This module acts as the "agent" layer:
   - Accepts a user query
   - Injects retrieved, relevant chunks as context
   - Prompts the LLM to generate a response grounded in that context

The LLM is explicitly instructed to rely only on provided context, enabling
transparent, auditable, and hallucination-resistant responses.
"""

from groq import Groq
from typing import List, Dict


class LLMClient:
    """Groq LLM client responsible for generating RAG-based responses."""

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        """
        Initialize the Groq LLM client.

        Args:
            api_key: API key used to authenticate with the Groq service.
            model: Name of the Groq-hosted language model to use.
        """
        self.client = Groq(api_key=api_key)
        self.model = model
        print(f"LLM client initialized with model: {model}")
