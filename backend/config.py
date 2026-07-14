"""Application configuration."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent
STORAGE_DIR = BASE_DIR / "storage"
UPLOAD_DIR = STORAGE_DIR / "uploads"
CORPUS_DIR = STORAGE_DIR / "corpus"
REPORT_DIR = STORAGE_DIR / "reports"
FAISS_DIR = STORAGE_DIR / "faiss_index"
DB_PATH = STORAGE_DIR / "plagiarism.db"

for directory in (STORAGE_DIR, UPLOAD_DIR, CORPUS_DIR, REPORT_DIR, FAISS_DIR):
    directory.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    """Application settings."""

    app_name: str = "AI Plagiarism Detection System"
    environment: Literal["development", "production"] = "development"
    log_level: str = "INFO"

    upload_dir: str = str(UPLOAD_DIR)
    corpus_dir: str = str(CORPUS_DIR)
    report_dir: str = str(REPORT_DIR)
    faiss_dir: str = str(FAISS_DIR)
    db_url: str = f"sqlite:///{DB_PATH}"

    chunk_size_sentences: int = 3
    chunk_overlap_sentences: int = 1

    embedding_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="SentenceTransformer model name.",
    )
    embedding_dim: int = 384
    embedding_batch_size: int = 32
    use_gpu_if_available: bool = True

    top_k: int = 5
    similarity_threshold: float = 0.55
    use_hybrid_retrieval: bool = True
    use_cross_encoder_rerank: bool = True
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    llm_provider: Literal["ollama", "openai", "none"] = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    llm_max_retries: int = 2
    llm_timeout_seconds: int = 30

    exact_copy_threshold: float = 0.97
    near_copy_threshold: float = 0.85
    paraphrase_threshold: float = 0.60

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list[str] = ["*"]

    class Config:
        env_file = ".env"
        env_prefix = "PLAG_"
        protected_namespaces = ("settings_",)


settings = Settings()