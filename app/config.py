"""
Configuration for RAG MVP using Pydantic.

This module defines application-level settings for the RAG MVP API,
including API configuration, document processing, vector store paths,
and RAG-specific parameters. It uses Pydantic's BaseSettings to
validate and manage environment variables. 

Derived properties provide environment-aware behavior, such as
debug mode toggling, vector store isolation, and safe retrieval
parameter adjustment. This ensures the RAG pipeline is consistent
and safe across development, staging, and production environments.
"""

from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    """Application settings"""
    
    # Environment Setting

    environment: Literal["development", "staging", "production"] = "development"

    # API Settings
    app_name: str = "RAG MVP API"
    debug: bool = True
    
    # Groq API
    groq_api_key: str
    groq_model: str = "llama-3.3-70b-versatile"  # Fast & free
    
    # Embedding Model
    embedding_model: str = "all-MiniLM-L6-v2"  # Fast, good quality
    
    # Document Processing
    chunk_size: int = 500
    chunk_overlap: int = 50
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    
    # Vector Store
    chroma_persist_dir: str = "./data/chroma_db"
    collection_name: str = "documents"
    
    # Paths
    upload_dir: str = "./data/uploads"
    
    # RAG Settings
    retrieval_top_k: int = 5


    # --------------------------------------
    # Derived / Environment-aware properties
    # -------------------------------------

    @property
    def is_production(self) -> bool:
        """Check if the current environment is production.

        Returns:
            bool: True if environment is 'production', False otherwise.
        """
        return self.environment == "production"


    @property
    def debug_mode(self) -> bool:
        """Determine if debug mode should be enabled based on environment.

        Debug is disabled in production and enabled in other environments.

        Returns:
            bool: True if debug mode should be active, False otherwise.
        """
        return not self.is_production


    @property
    def chroma_dir(self) -> str:
        """Get the environment-specific vector store directory.

        Appends the environment name to the base Chroma directory to
        prevent dev/prod data contamination.

        Returns:
            str: Full path to the environment-specific Chroma DB directory.
        """
        return f"{self.chroma_persist_dir}_{self.environment}"

    @property
    def effective_top_k(self) -> int:
        """Compute a safe top-k value for document retrieval based on environment.

        In development, limits top_k to 3 for faster iteration. In production,
        uses the configured retrieval_top_k value.

        Returns:
            int: Number of top document chunks to retrieve.
        """
        if self.environment == "development":
            return min(self.retrieval_top_k, 3)
        return self.retrieval_top_k

    class Config:
        """Pydantic configuration for environment loading and case sensitivity."""
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()

# Ensure directories exist
Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
