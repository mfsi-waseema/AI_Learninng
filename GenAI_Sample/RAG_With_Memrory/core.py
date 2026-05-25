"""
Runtime helpers for the Streamlit app — independent of the stages/ folder.

Stage files are teaching material; this module is the production-style code
that app.py actually runs. It centralises:
  - Qdrant client + Gemini embeddings + Groq chat LLM
  - Two-stage retrieval (vector search + BGE cross-encoder rerank)
  - The three @tool functions the agent can call
  - The AgentExecutor builder (LangChain tool-calling agent)
  - Short memory: SQL-backed chat history (SQLAlchemy / SQLite)
  - Long memory: full chat persisted in Qdrant (QdrantChatMemory)

Heavy resources (cross-encoder, vectorstore, embeddings client) are cached
with `lru_cache` so they load once per process.
"""
from __future__ import annotations

import ast
import operator
import time
import uuid
from datetime import datetime, timezone
from functools import lru_cache

import bs4
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Optional
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from config import settings

SAMPLE_URLS = [
    "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
    "https://en.wikipedia.org/wiki/Vector_database",
]


# ---------- clients (cached singletons) -------------------------------------

@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

@lru_cache(maxsize=1)
def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.google_api_key,
    )


@lru_cache(maxsize=1)
def _get_cross_encoder() -> HuggingFaceCrossEncoder:
    """Load the configured reranker once per process.

    Default: cross-encoder/ms-marco-MiniLM-L6-v2 (~80 MB, fast, English-only).
    Swap to BAAI/bge-reranker-base (~1.1 GB) for higher quality / multilingual.
    """
    return HuggingFaceCrossEncoder(model_name=settings.reranker_model)


def get_chat_llm(temperature: float = 0.2) -> ChatGroq:
    """Chat / agent LLM. Groq's free tier is generous and fast.

    Embeddings stay on Gemini (Groq doesn't offer an embeddings endpoint).
    `disable_streaming=True` avoids a Groq quirk where some open-weights
    models occasionally emit malformed tool-call JSON mid-stream.
    """
    return ChatGroq(
        model=settings.chat_model,
        groq_api_key=settings.groq_api_key,
        temperature=temperature,
        disable_streaming=True,
    )


@lru_cache(maxsize=4)
def get_vectorstore(collection: str | None = None) -> QdrantVectorStore:
    return QdrantVectorStore(
        client=get_qdrant_client(),
        collection_name=collection or settings.docs_collection,
        embedding=get_embeddings(),
    )


# ---------- one-time bootstrap ----------------------------------------------

def bootstrap_qdrant() -> None:
    """Ensure both Qdrant collections exist at the configured embedding dim.

    If a collection exists with the wrong dim, drop and recreate it.
    Safe to call repeatedly; no-op when collections are already correct.
    """
    client = get_qdrant_client()
    existing = {c.name: c for c in client.get_collections().collections}
    for name in (settings.docs_collection, settings.chat_memory_collection):
        if name in existing:
            info = client.get_collection(name)
            if info.config.params.vectors.size == settings.embedding_dim:
                continue
            client.delete_collection(name)
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=settings.embedding_dim, distance=Distance.COSINE
            ),
        )


# ---------- ingestion -------------------------------------------------------

def load_and_chunk(urls: list[str] | None = None):
    
    """Fetch web pages, strip noise, split into ~1000-char chunks (200-char overlap)."""
    loader = WebBaseLoader(
        web_paths=urls or SAMPLE_URLS,
        bs_kwargs={"parse_only": bs4.SoupStrainer(["p", "h1", "h2", "h3", "li"])},
    )
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)


def ingest_urls(urls: list[str]) -> int:
    """Load + chunk + index into the docs collection. Returns chunk count."""
    chunks = load_and_chunk(urls)
    get_vectorstore().add_documents(chunks)
    return len(chunks)


# ---------- short memory (SQLAlchemy) ---------------------------------------

def get_sql_chat_history(session_id: str) -> SQLChatMessageHistory:
    """SQLAlchemy-backed chat history (SQLite by default).

    Swap CHAT_HISTORY_DB in .env to a Postgres / MySQL connection string
    and the same code stores history there instead — that's the SQLAlchemy
    payoff.
    """
    return SQLChatMessageHistory(
        session_id=session_id,
        connection=settings.chat_history_db,
        table_name="chat_messages",
    )


def list_chat_sessions() -> list[dict]:
    """Return all known chat sessions, newest activity first.

    Reads `chat_messages` (the SQLAlchemy table our SQLChatMessageHistory uses).
    Returns [{"session_id": str, "message_count": int}, ...]. If the table
    doesn't exist yet (no chats ever saved), returns [].
    """
    import sqlalchemy as sa

    engine = sa.create_engine(settings.chat_history_db)
    inspector = sa.inspect(engine)
    if "chat_messages" not in inspector.get_table_names():
        return []
    sql = sa.text(
        "SELECT session_id, COUNT(*) AS n, MAX(id) AS last_id "
        "FROM chat_messages GROUP BY session_id ORDER BY last_id DESC"
    )
    with engine.connect() as conn:
        rows = conn.execute(sql).fetchall()
    return [{"session_id": r[0], "message_count": r[1]} for r in rows]


def load_session_messages(session_id: str) -> list[dict]:
    """Reconstruct UI-ready messages from the SQL chat history for a session."""
    history = get_sql_chat_history(session_id)
    out: list[dict] = []
    for m in history.messages:
        role = "user" if m.type == "human" else "assistant"
        out.append({"role": role, "content": m.content})
    return out


# ---------- two-stage retrieval (vector + cross-encoder rerank) -------------

# Toggled by the Streamlit UI to compare plain vs reranked retrieval.
_RERANKER_ENABLED = True


def set_reranker_enabled(enabled: bool) -> None:
    """UI hook: turn the cross-encoder reranker on/off at runtime.

    When OFF, search_docs uses plain vector similarity (no rerank) so you can
    see the quality difference side-by-side.
    """
    global _RERANKER_ENABLED
    _RERANKER_ENABLED = enabled


def is_reranker_enabled() -> bool:
    return _RERANKER_ENABLED


@lru_cache(maxsize=8)
def build_reranked_retriever(top_k_initial: int = 20, top_n_final: int = 5):
    """Vector search top-K, then cross-encoder picks top-N.

    Cached per (top_k_initial, top_n_final) so the cross-encoder loads once.
    """
    base = get_vectorstore().as_retriever(search_kwargs={"k": top_k_initial})
    reranker = CrossEncoderReranker(model=_get_cross_encoder(), top_n=top_n_final)
    return ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base,
    )


def build_plain_retriever(k: int = 5):
    """Vector similarity only, no reranker. For toggle-off comparison."""
    return get_vectorstore().as_retriever(search_kwargs={"k": k})


def get_active_retriever(k_initial: int = 20, k_final: int = 5):
    """Return whichever retriever the UI toggle currently selects."""
    if _RERANKER_ENABLED:
        return build_reranked_retriever(top_k_initial=k_initial, top_n_final=k_final)
    return build_plain_retriever(k=k_final)


# ---------- agent tools -----------------------------------------------------

@tool
def search_docs(query: str) -> str:
    """Search the indexed documents for information relevant to the query.
    Use this whenever the user asks a factual or knowledge question."""
    retriever = get_active_retriever(k_initial=20, k_final=4)
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant documents found."
    return "\n\n".join(
        f"[Source: {d.metadata.get('source')}]\n{d.page_content}" for d in docs
    )


_SAFE_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_SAFE_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_BINOPS:
        return _SAFE_BINOPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_UNARYOPS:
        return _SAFE_UNARYOPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Unsupported expression")


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression like '12 * (3 + 4)' and return the result.
    Supports + - * / % ** and parentheses on numbers only."""
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_safe_eval(tree))
    except Exception as e:
        return f"Error: {e}"


@tool
def current_time() -> str:
    """Return the current UTC date and time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ----- a STRUCTURED tool (multi-arg, Pydantic schema) -----------------------
# The single-arg @tool functions above auto-generate a one-field schema from
# the function signature. For multi-arg tools, defining the schema as a
# Pydantic class makes field descriptions, defaults, and bounds explicit —
# the LLM reads those descriptions to decide what to pass.

class AdvancedSearchInput(BaseModel):
    """Args for `search_docs_advanced`."""

    query: str = Field(description="The search query in natural language.")
    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description="How many chunks to return. Use 1-3 for a focused answer, 5-10 for broad context.",
    )
    source_contains: Optional[str] = Field(
        default=None,
        description=(
            "Optional substring to filter results by source URL. "
            "Examples: 'wikipedia.org', 'arxiv.org'. Leave empty for no filter."
        ),
    )


@tool(args_schema=AdvancedSearchInput)
def search_docs_advanced(
    query: str,
    top_k: int = 5,
    source_contains: Optional[str] = None,
) -> str:
    """Search the indexed docs with controllable result count and source filter.
    Use this instead of search_docs when the user asks for results from a
    specific site, or wants more / fewer chunks than the default."""
    retriever = get_active_retriever(k_initial=20, k_final=top_k)
    docs = retriever.invoke(query)
    if source_contains:
        docs = [d for d in docs if source_contains in (d.metadata.get("source") or "")]
    if not docs:
        return "No relevant documents found."
    return "\n\n".join(
        f"[Source: {d.metadata.get('source')}]\n{d.page_content}" for d in docs
    )


ALL_TOOLS = [search_docs, search_docs_advanced, calculator, current_time]


# ---------- agent builder ---------------------------------------------------

AGENT_SYSTEM_PROMPT = (
    "You are a helpful research assistant.\n"
    "Tool selection:\n"
    "- search_docs(query): default knowledge lookup. Use for any factual question.\n"
    "- search_docs_advanced(query, top_k, source_contains): use when the user "
    "asks for more/fewer results, or restricts to a specific source "
    "(e.g. 'from wikipedia', 'just one result').\n"
    "- calculator(expression): for arithmetic.\n"
    "- current_time(): for date / time questions.\n"
    "If a search doesn't have enough information, search again with a better query. "
    "Always cite sources when you used a search tool."
)


def build_agent_executor(
    extra_system_messages: list[str] | None = None,
    verbose: bool = True,
) -> AgentExecutor:
    """Build a LangChain tool-calling AgentExecutor with the standard tools.

    extra_system_messages: optional extra system-level context to prepend
        (e.g., recalled long-term memories from Qdrant).
    """
    system_chunks = [AGENT_SYSTEM_PROMPT]
    if extra_system_messages:
        system_chunks.extend(extra_system_messages)
    system_text = "\n\n".join(system_chunks)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_text),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(get_chat_llm(), ALL_TOOLS, prompt)
    return AgentExecutor(agent=agent, tools=ALL_TOOLS, verbose=verbose)


# ---------- long memory (Qdrant, full chat persisted) -----------------------

class QdrantChatMemory:
    """Persist full chat turns in a Qdrant collection.

    - save_turn(): embeds + upserts user message AND assistant reply as two
      separate points. Full text stored in payload.
    - recall(): vector-searches for past turns most relevant to a query,
      optionally filtered by session_id.
    """

    def __init__(self) -> None:
        self.client = get_qdrant_client()
        self.embeddings = get_embeddings()
        self.collection = settings.chat_memory_collection

    def save_turn(self, session_id: str, user_msg: str, assistant_msg: str) -> None:
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        turn_index = int(time.time() * 1000)
        texts = [user_msg, assistant_msg]
        roles = ["user", "assistant"]
        vectors = self.embeddings.embed_documents(texts)
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={
                    "session_id": session_id,
                    "role": role,
                    "text": text,
                    "timestamp": ts,
                    "turn_index": turn_index,
                },
            )
            for vec, role, text in zip(vectors, roles, texts)
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def recall(
        self,
        query: str,
        k: int = 3,
        session_id: str | None = None,
    ) -> list[dict]:
        vec = self.embeddings.embed_query(query)
        flt: Filter | None = None
        if session_id is not None:
            flt = Filter(
                must=[FieldCondition(key="session_id", match=MatchValue(value=session_id))]
            )
        result = self.client.query_points(
            collection_name=self.collection,
            query=vec,
            query_filter=flt,
            limit=k,
        )
        return [
            {
                "score": h.score,
                "role": h.payload.get("role"),
                "text": h.payload.get("text"),
                "timestamp": h.payload.get("timestamp"),
                "session_id": h.payload.get("session_id"),
            }
            for h in result.points
        ]

    def format_for_prompt(self, recalled: list[dict]) -> str:
        if not recalled:
            return ""
        lines = ["Relevant past conversations (recalled from long-term memory):"]
        for r in recalled:
            lines.append(f"  ({r['role']} @ {r['timestamp']}) {r['text']}")
        return "\n".join(lines)
