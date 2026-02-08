"""
Document ingestion and preprocessing for the RAG pipeline.

This module is responsible for converting raw PDF documents into
structured, retrieval-ready text chunks. It represents the first
stage of the RAG pipeline, where unstructured source documents are
validated, normalized, and transformed into deterministic text
segments suitable for embedding and storage in a vector database.

At this stage, the system:
- Extracts raw text from PDF files
- Splits text into overlapping chunks optimized for retrieval
- Generates stable document identifiers
- Produces metadata required by downstream indexing and retrieval steps
"""



from typing import List, Dict, Any
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
import hashlib



class DocumentProcessor:
    """Processes PDF documents into text chunks for retrieval.

    This class encapsulates the document ingestion logic for the RAG
    pipeline. It ensures that raw documents are consistently converted
    into chunked text representations that can be embedded, indexed,
    and retrieved by downstream components.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """Initialize the document processor.

        Args:
            chunk_size (int): Maximum number of characters per chunk.
            chunk_overlap (int): Number of overlapping characters
                between consecutive chunks to preserve context.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    

