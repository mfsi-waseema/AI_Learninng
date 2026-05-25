# Agentic RAG Demo (LangChain + Qdrant + Reranking + Memory)

## What's covered

- **Vector DB (Qdrant)** running in Docker
- **Loading + chunking** web pages
- **Embeddings** with Google Gemini (`gemini-embedding-001`, 3072 dim)
- **Basic similarity search**
- **Prompts + LCEL chains** (`prompt | llm | parser`)
- **Structured output** (`with_structured_output` + Pydantic)
- **Reranking** with a local cross-encoder (`BAAI/bge-reranker-base`)
- **Custom tools** (`@tool` decorator)
- **Agentic RAG** (`create_tool_calling_agent` + `AgentExecutor`)
- **Short memory** persisted via **SQLAlchemy** (`SQLChatMessageHistory` over SQLite — swap conn-string for Postgres/MySQL)
- **Long memory** — full chat persisted in Qdrant for cross-session semantic recall
- **Observability** (`set_debug` + `verbose=True`)
- **Streamlit chat UI** wiring everything together

## Stack

| Layer | Used here | Why |
|---|---|---|
| Embeddings | Google Gemini `gemini-embedding-001` | Generous free tier on the embeddings quota; 3072-dim vectors |
| Chat / Agent | Groq `openai/gpt-oss-20b` | Generous free tier; reliable tool-calling on Groq |
| Vector DB | Qdrant (Docker) | Fast, runs locally, on-disk persistence |
| Reranker | `cross-encoder/ms-marco-MiniLM-L6-v2` (local) | Free, runs on CPU, no API key. ~80 MB download once. Swap via `RERANKER_MODEL` |
| Short memory | SQLite via SQLAlchemy | Survives restarts; same code works against Postgres/MySQL |
| Long memory | Qdrant `chat_memory` collection | Semantic recall across sessions |
| UI | Streamlit | Minimal, fast to iterate |

## One-time setup

```bash
cd /Users/waseem/Data/Study/Python/GEN-AI/app

# 1. Python venv
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 2. Env vars
cp .env.example .env
# then edit .env and fill in:
#   GOOGLE_API_KEY  (https://aistudio.google.com/app/apikey)
#   GROQ_API_KEY    (https://console.groq.com/keys)

# 3. Start Qdrant in Docker (Docker Desktop must be running first)
docker compose up -d
# verify: open http://localhost:6333/dashboard
```

## Run the final app

```bash
source venv/bin/activate
streamlit run app.py
```

The first time:

1. The app auto-creates the `docs` and `chat_memory` collections in Qdrant
2. The reranker model downloads to `~/.cache/huggingface/hub/` on the first chat turn (~80 MB for the default MiniLM, ~1.1 GB if you switched to BGE)
3. Subsequent turns are 2–5 seconds (model is cached)

In the UI:

- **Sidebar:** paste URLs (defaults: two Wikipedia pages on RAG and vector DBs) → click **Ingest** → chunks + embeds + indexes them
- **Main:** type a question. Each answer has two expandable panels:
  - **Recalled long-term memories** — past chat turns Qdrant matched semantically
  - **Retrieved chunks (reranked)** — document chunks the cross-encoder picked

Restart the app, start a new session, ask "what did we discuss about X earlier?"
The long-memory recall still finds it because Qdrant is on disk.


## Project layout

```
app/
├── app.py                      # Streamlit UI — standalone, imports only core + config
├── core.py                     # runtime helpers, agent builder, QdrantChatMemory, tools
├── config.py                   # pydantic-settings, reads .env
├── docker-compose.yml          # Qdrant service
├── requirements.txt
├── .env.example                # template — copy to .env and fill keys
├── chat_history.db             # SQLite short memory (gitignored, auto-created)
├── qdrant_storage/             # Qdrant data dir (gitignored, mounted into container)
```

## Architecture (per chat turn)

```
user question
  │
  ├─► QdrantChatMemory.recall(question, k=3)      ── long memory: semantic search across all past turns
  │
  ├─► AgentExecutor (Groq gpt-oss-20b + tools)
  │     │
  │     ├─► chat_history (last N turns from SQLite via SQLAlchemy)  ── short memory
  │     │
  │     └─► may call any of:
  │           • search_docs(query)   ──► reranked retriever (Qdrant top-20 → BGE top-5)
  │           • calculator(expr)
  │           • current_time()
  │
  ├─► QdrantChatMemory.save_turn(session_id, user, assistant)   ── persist this turn
  │
  └─► render answer + Recalled memories + Retrieved chunks panels
```

## Configuration (`.env`)

| Variable | What it does |
|---|---|
| `GOOGLE_API_KEY` | For Gemini embeddings only |
| `GROQ_API_KEY` | For chat / agent (Groq) |
| `QDRANT_HOST`, `QDRANT_PORT` | Defaults to `localhost:6333` |
| `DOCS_COLLECTION` | Qdrant collection for indexed documents |
| `CHAT_MEMORY_COLLECTION` | Qdrant collection for long-term chat memory |
| `EMBEDDING_MODEL` | Default `models/gemini-embedding-001` (3072 dim) |
| `CHAT_MODEL` | Default `openai/gpt-oss-20b`. Other Groq options: `openai/gpt-oss-120b`, `meta-llama/llama-4-scout-17b-16e-instruct`, `qwen/qwen3-32b`, `llama-3.1-8b-instant` |
| `RERANKER_MODEL` | Default `cross-encoder/ms-marco-MiniLM-L6-v2` (80 MB, fast, English). Higher quality alternative: `BAAI/bge-reranker-base` (1.1 GB, multilingual) |
| `CHAT_HISTORY_DB` | SQLAlchemy connection string — default `sqlite:///chat_history.db`. Swap for `postgresql://...` etc. |

To change the LLM, just edit `CHAT_MODEL` in `.env` and restart the app.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Cannot connect to the Docker daemon` | Start Docker Desktop, then `docker compose up -d` |
| `ConnectionError` to localhost:6333 | Qdrant container isn't up. Run `docker compose ps` |
| `ValidationError: GOOGLE_API_KEY` / `GROQ_API_KEY` | `.env` is missing or has placeholder values |
| Empty results in stages 04+ | You skipped stage 03 — no chunks indexed yet (or run the Streamlit Ingest button) |
| Stage 08 / first app turn pauses | One-time reranker model download (default ~80 MB MiniLM, or ~1.1 GB if you set BGE). Check terminal for progress |
| `429 ResourceExhausted` from Gemini | Free-tier quota hit. Embeddings have a separate, much larger quota — usually only chat hits this. (Chat is now Groq, not Gemini, so this is unlikely) |
| `groq.APIError: Failed to call a function` | Some Groq models emit malformed tool-call JSON. We disable streaming + use `openai/gpt-oss-20b`, which fixes it. If you change `CHAT_MODEL`, pick a tool-calling-friendly one |
| Streamlit chat hangs forever on first turn | BGE model still downloading or loading. Watch the terminal — once `model.safetensors` shows 100%, the next turn finishes |
