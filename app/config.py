"""
Configuration for RAG MVP
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
        return self.environment == "production"

    @property
    def debug_mode(self) -> bool:
        return not self.is_production

    @property
    def chroma_dir(self) -> str:
        """Isolate vector DB per environment"""
        return f"{self.chroma_persist_dir}_{self.environment}"

    @property
    def effective_top_k(self) -> int:
        """Faster iteration in dev, higher recall in prod"""
        if self.environment == "development":
            return min(self.retrieval_top_k, 3)
        return self.retrieval_top_k

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()

# Ensure directories exist
Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
