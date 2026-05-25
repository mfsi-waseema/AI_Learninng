from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    google_api_key: str
    groq_api_key: str
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    docs_collection: str = "docs"
    chat_memory_collection: str = "chat_memory"
    embedding_model: str = "models/gemini-embedding-001"
    chat_model: str = "openai/gpt-oss-20b"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    embedding_dim: int = 3072
    chat_history_db: str = "sqlite:///chat_history.db"


settings = Settings()
