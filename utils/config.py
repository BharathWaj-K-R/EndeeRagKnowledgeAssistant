"""Configuration helpers for the project."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    app_name: str = "AI Knowledge Assistant"
    upload_dir: str = str(BASE_DIR / "data" / "uploads")
    chroma_dir: str = str(BASE_DIR / "data" / "chroma")
    chroma_collection_name: str = "knowledge_base"
    embedding_model_name: str = "all-MiniLM-L6-v2"
    embedding_provider: str = "sentence_transformers"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = "gpt-4o-mini"
    chunk_size: int = 700
    chunk_overlap: int = 120
    similarity_threshold: float | None = 0.25

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once and ensure runtime directories exist."""
    settings = Settings()
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.chroma_dir).mkdir(parents=True, exist_ok=True)
    return settings
