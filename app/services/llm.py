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

    def generate_rag_response(
        self,
        query: str,
        context_chunks: List[Dict]
    ) -> str:
        """
        Generate a response using Retrieval-Augmented Generation (RAG).

        This method combines a user query with retrieved document chunks
        (semantic context) and sends a structured prompt to the LLM.
        The model is instructed to answer strictly based on the provided
        context and to acknowledge when insufficient information is available.

        Args:
            query: The user's natural language question.
            context_chunks: Retrieved document chunks from the vector store,
                including text content and metadata.

        Returns:
            A generated answer grounded in the retrieved context.

        Raises:
            Exception: If the Groq API request fails.
        """
        # Build readable context from retrieved chunks
        context = self._build_context(context_chunks)

        # Create the RAG prompt
        prompt = self._create_rag_prompt(query, context)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant that answers questions "
                            "based on the provided context. If the answer is not "
                            "contained in the context, say so clearly."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0.3,
                max_tokens=1024,
            )

            return response.choices[0].message.content

        except Exception as e:
            raise Exception(f"Groq API error: {str(e)}")
        

    def _build_context(self, chunks: List[Dict]) -> str:
        """
        Build a formatted context string from retrieved document chunks.

        Each chunk is labeled with a source index and filename to improve
        traceability and interpretability of the generated response.

        Args:
            chunks: Retrieved chunks containing text content and metadata.

        Returns:
            A formatted string representing the full retrieval context.
        """
        if not chunks:
            return "No relevant context found."

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            filename = chunk["metadata"].get("filename", "Unknown")
            content = chunk["content"]
            context_parts.append(
                f"[Source {i} - {filename}]\n{content}"
            )

        return "\n\n".join(context_parts)
    

    def _create_rag_prompt(self, query: str, context: str) -> str:
        """
        Create a structured RAG prompt combining context and query.

        The prompt explicitly instructs the model to rely only on the provided
        context when generating an answer, reducing hallucinations and improving
        trustworthiness.

        Args:
            query: The user's original question.
            context: Aggregated context from retrieved document chunks.

        Returns:
            A formatted prompt string ready for LLM consumption.
        """
        return f"""Based on the following context, please answer the question.
If the answer is not in the context, say "I don't have enough information to answer this question."

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""