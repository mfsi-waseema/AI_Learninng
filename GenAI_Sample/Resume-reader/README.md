# Smart Research Assistant

A RAG (Retrieval-Augmented Generation) system that answers questions from your PDFs using Gemini and vector search.

## Tech Stack

- **Python 3.11**
- **FastAPI** + **Uvicorn** - REST API
- **Streamlit** - Frontend UI
- **LangChain** + **Google Gemini** - LLM & Embeddings
- **Qdrant** - Vector Database
- **PyPDF** - PDF text extraction
- **Docker** + **Docker Compose** - Containerization

## File Structure

```
smart-research-assistant/
├── app/
│   ├── main.py                 # App entry point
│   ├── starter.py              # App factory, router registration
│   ├── health.py               # Health check endpoint
│   ├── agents/
│   │   └── research_agent.py   # Retrieval + prompt + LLM orchestration
│   ├── config/
│   │   ├── settings.py         # Environment settings
│   │   ├── logging.py          # JSON structured logging
│   │   └── qdrant.py           # Qdrant connection setup
│   ├── constants/              # App constants (future)
│   ├── exceptions/             # Domain exceptions (future)
│   ├── llms/
│   │   └── gemini_client.py    # Gemini streaming client
│   ├── models/                 # SQLAlchemy models (future)
│   ├── prompts/
│   │   └── templates.py        # System & user prompt templates
│   ├── repository/             # Data access layer (future)
│   ├── routes/
│   │   ├── core_routes/        # Chat & ingestion endpoints (future)
│   │   └── iam_routes/         # Auth & user endpoints (future)
│   ├── schemas/                # Pydantic request/response (future)
│   ├── services/               # Business logic layer (future)
│   ├── tools/
│   │   └── retriever.py        # Embedding + vector search
│   ├── utils/
│   │   └── core_utils/
│   │       └── guard.py        # Prompt injection checks
│   ├── workers/                # Background workers (future)
│   └── tests/                  # Pytest tests (future)
├── ui/
│   └── app.py                  # Streamlit frontend
├── scripts/
│   ├── ingest_qdrant.py        # PDF ingestion script
│   └── sample_resume.pdf       # Sample PDF for demo
├── .env                        # Environment variables
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

User types question
      |
  [Streamlit UI] ---HTTP---> [FastAPI /ask]
                                  |
                          [Safety Guard] --blocked?--> "Request blocked"
                                  |
                          [Embed question into vector]
                                  |
                          [Search Qdrant for similar chunks]
                                  |
                          [Build prompt: system + context + question]
                                  |
                          [Stream answer from Gemini]
                                  |
                    <---tokens stream back to UI---

## Run Commands

### 1. Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Add your API key to `.env`:

```env
GOOGLE_API_KEY=your_google_ai_studio_key
```

### 2. Run with Docker

```bash
docker compose up --build
docker compose exec api python scripts/ingest_qdrant.py
streamlit run ui/app.py
```

### 3. Run Locally (without Docker API)

```bash
docker compose up qdrant
python scripts/ingest_qdrant.py
uvicorn app.main:app --reload
streamlit run ui/app.py
```

### Useful URLs

- Swagger UI: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`
- Streamlit UI: `http://localhost:8501`
