import os
from dataclasses import dataclass

from dotenv import load_dotenv

from app.prompts.templates import RESUME_READER_SYSTEM_PROMPT

load_dotenv()


@dataclass(frozen=True)
class Settings:
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    system_prompt: str = os.getenv("SYSTEM_PROMPT", RESUME_READER_SYSTEM_PROMPT)
    gemini_chat_model: str = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")
    gemini_embedding_model: str = os.getenv(
        "GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001"
    )
    qdrant_host: str = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port: int = int(os.getenv("QDRANT_PORT", "6333"))
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "docs")
    rag_top_k: int = int(os.getenv("RAG_TOP_K", "3"))
    api_base_url: str = os.getenv("API_BASE_URL", "http://localhost:8000")


settings = Settings()
