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

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract all textual content from a PDF file.

        Args:
            pdf_path (str): Path to the PDF file.

        Returns:
            str: Extracted text content from all pages.

        Raises:
            ValueError: If text extraction fails.
        """
        try:
            reader = PdfReader(pdf_path)
            text = ""

            for page in reader.pages:
                text += page.extract_text() + "\n"

            return text.strip()

        except Exception as exc:
            raise ValueError(
                f"Failed to extract text from PDF: {str(exc)}"
            ) from exc
        

    def chunk_text(self, text: str) -> List[str]:
        """Split raw text into overlapping chunks.

        Chunking improves retrieval quality by balancing semantic
        completeness with embedding efficiency.

        Args:
            text (str): Full extracted document text.

        Returns:
            List[str]: List of text chunks.
        """
        return self.splitter.split_text(text)
    

    def process_document(
        self,
        file_path: str,
        filename: str
    ) -> Dict[str, Any]:
        """Process a PDF document into structured chunks and metadata.

        This method orchestrates the full ingestion flow:
        extraction → validation → chunking → metadata generation.

        Args:
            file_path (str): Path to the PDF file on disk.
            filename (str): Original uploaded filename.

        Returns:
            Dict[str, Any]: A structured representation of the document
            containing:
                - doc_id (str): Stable document identifier
                - filename (str): Original filename
                - chunks (List[str]): Text chunks for embedding
                - metadata (Dict): Document-level metadata

        Raises:
            ValueError: If the PDF contains insufficient or no text.
        """
        doc_id = self._generate_doc_id(file_path)

        text = self.extract_text_from_pdf(file_path)

        if not text or len(text) < 10:
            raise ValueError(
                "PDF appears to be empty or has no extractable text"
            )

        chunks = self.chunk_text(text)

        metadata = {
            "filename": filename,
            "total_chars": len(text),
            "num_chunks": len(chunks),
        }

        return {
            "doc_id": doc_id,
            "filename": filename,
            "chunks": chunks,
            "metadata": metadata,
        }
    
    def _generate_doc_id(self, file_path: str) -> str:
        """Generate a stable document ID based on file content.

        Using a content hash ensures deterministic IDs and prevents
        duplicate indexing of identical documents.

        Args:
            file_path (str): Path to the PDF file.

        Returns:
            str: Deterministic document identifier.
        """
        with open(file_path, "rb") as file:
            file_hash = hashlib.md5(file.read()).hexdigest()

        return f"doc_{file_hash[:12]}"
    
