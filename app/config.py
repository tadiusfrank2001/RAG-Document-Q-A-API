"""
Configuration for RAG MVP
"""
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    """Application settings"""
    
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
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()

# Ensure directories exist
Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
