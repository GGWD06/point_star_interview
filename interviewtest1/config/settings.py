"""
Centralized configuration via Pydantic BaseSettings.
All values are loaded from environment variables / .env file.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- LLM ---
    google_api_key: str = ""
    google_model_classifier: str = "gemini-flash-latest"
    google_model_drafter: str = "gemini-flash-latest"
    google_model_judge: str = "gemini-flash-latest"
    google_embedding_model: str = "models/gemini-embedding-2"

    # --- Vector Store ---
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "email_support_kb"

    # --- RAG ---
    rag_top_k: int = 5
    rag_relevance_threshold: float = 0.7

    # --- Redis ---
    redis_url: str = ""

    # --- Escalation ---
    escalation_contact_window_days: int = 7
    escalation_contact_threshold: int = 3

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_key: str = "change-me-in-production"


# Singleton instance — import this everywhere
settings = Settings()
