# FastAPI Task Manager API

A beginner-friendly Task Manager API built with FastAPI and in-memory storage.

## Features

- Create a task
- Get all tasks
- Get one task by ID
- Update a task
- Delete a task
- Optional filtering by completion (`?completed=true` or `?completed=false`)

## Task Model

- `id` (int)
- `title` (str)
- `description` (str, optional)
- `completed` (bool, default `false`)

## 1) Run Locally with Virtual Environment

### Create and activate virtual env

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Start the API

```bash
uvicorn main:app --reload
```

The API runs on: `http://127.0.0.1:8000`

## 2) Run Streamlit Frontend

Open a second terminal (keep FastAPI running), then run:

```bash
streamlit run streamlit_app.py
```

The Streamlit UI runs on: `http://localhost:8501`

## 3) API Docs

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

## 4) Run with Docker

### Build image

```bash
docker build -t task-manager-api .
```

### Run container

```bash
docker run --rm -p 8000:8000 task-manager-api
```

## 5) Run with Docker Compose

```bash
docker compose up --build
```

## 6) Example Requests (curl)

### Create task

```bash
curl -X POST "http://127.0.0.1:8000/tasks" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Learn FastAPI",
    "description": "Build a beginner project",
    "completed": false
  }'
```

### Get all tasks

```bash
curl "http://127.0.0.1:8000/tasks"
```

### Filter completed tasks

```bash
curl "http://127.0.0.1:8000/tasks?completed=true"
```

### Get task by ID

```bash
curl "http://127.0.0.1:8000/tasks/1"
```

### Update task

```bash
curl -X PUT "http://127.0.0.1:8000/tasks/1" \
  -H "Content-Type: application/json" \
  -d '{
    "completed": true
  }'
```

### Delete task

```bash
curl -X DELETE "http://127.0.0.1:8000/tasks/1"
```
