from app.config.qdrant import get_client, COLLECTION
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config.settings import settings
from app.config.logging import get_logger, log_event

logger = get_logger("rag.retriever")
embeddings = GoogleGenerativeAIEmbeddings(
    model=settings.gemini_embedding_model,
    google_api_key=settings.google_api_key,
)


def embed(query: str):
    return embeddings.embed_query(query)


async def retrieve_docs(query: str):
    client = get_client()
    vector = embed(query)
    result = client.query_points(
        collection_name=COLLECTION,
        query=vector,
        limit=settings.rag_top_k,
    )
    hits = result.points if hasattr(result, "points") else []
    docs = [h.payload.get("text", "") for h in hits if h.payload]
    scores = [getattr(h, "score", None) for h in hits]
    log_event(logger, "retrieval_complete", query=query, docs_count=len(docs), scores=scores)
    return docs
