import os
from pydantic import BaseSettings, Field, validator
from pathlib import Path

class Settings(BaseSettings):
    # Paths
    RAW_PDF_PATH: str = Field(default="data/raw_pdfs", description="Directory with raw PDF files")
    CHROMA_DB_PATH: str = Field(default="data/chroma_db", description="Directory for persistent Chroma collection")

    # Embedding model configuration
    EMBEDDING_MODEL: str = Field(default="BAAI/bge-small-en-v1.5", description="Local sentence‑transformers model name")
    OPENAI_EMBEDDING_MODEL: str = Field(default="", description="OpenAI embedding model name if used")

    # LLM configuration
    LLM_PROVIDER: str = Field(default="ollama", description="Provider: ollama, openai, or anthropic")
    LLM_MODEL: str = Field(default="llama3.1", description="Model name for the selected provider")
    OPENAI_API_KEY: str = Field(default="", env="OPENAI_API_KEY")
    ANTHROPIC_API_KEY: str = Field(default="", env="ANTHROPIC_API_KEY")
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434", description="Base URL for Ollama server")

    # Retrieval settings
    TOP_K: int = Field(default=5, ge=1, description="Number of chunks to retrieve per query")
    HYBRID_DENSE_WEIGHT: float = Field(default=0.7, ge=0.0, le=1.0, description="Weight for dense vs. sparse retrieval")

    # Chunking settings
    CHUNK_SIZE: int = Field(default=1000, description="Maximum characters per chunk")
    CHUNK_OVERLAP: int = Field(default=200, description="Number of overlapping characters between chunks")

    # LangSmith tracing
    LANGCHAIN_TRACING_V2: bool = Field(default=False, description="Enable LangSmith tracing when true")
    LANGCHAIN_API_KEY: str = Field(default="", env="LANGCHAIN_API_KEY")
    LANGCHAIN_PROJECT: str = Field(default="rag-qa-project", description="LangSmith project name")

    @validator("RAW_PDF_PATH", "CHROMA_DB_PATH", pre=True)
    def resolve_path(cls, v):
        # Resolve relative to project root (directory containing this file's parent)
        return str(Path(__file__).parents[2] / v)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Export a singleton for easy import
settings = Settings()
