# DocuMind — RAG-Based Internal Knowledge Assistant

Ask natural language questions over your own PDF, Word, and text documents.
Answers are grounded in retrieved context with source citations — no hallucinations.

## Architecture

```
User → Streamlit UI → FastAPI Backend → Vector Store (FAISS/ChromaDB)
                                      ↓
                              ConversationalRetrievalChain (MMR)
                                      ↓
                              OpenAI LLM → Cited Answer
```

## Features

- Multi-format ingestion: PDF, DOCX, TXT
- Overlapping chunks for boundary-safe retrieval
- MMR (Maximal Marginal Relevance) for diverse, non-redundant context
- Conversation memory — follow-up questions work naturally
- Source citations per response
- Switchable vector store: FAISS (ephemeral) or ChromaDB (persistent)
- Fully Dockerized — one command to run

## Setup

### Local

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env       # set OPENAI_API_KEY
uvicorn backend.main:app --reload &
streamlit run frontend/app.py
```

### Docker

```bash
cp .env.example .env       # set OPENAI_API_KEY
docker-compose up --build
# Frontend: http://localhost:8501
# API docs: http://localhost:8000/docs
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/upload` | Upload & ingest a document |
| POST | `/query` | Ask a question |
| DELETE | `/reset` | Clear store & session |
| POST | `/session/new` | Create a new session ID |
| GET | `/health` | Health check |

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | Required |
| `VECTOR_STORE` | `faiss` | `faiss` or `chroma` |
| `CHUNK_SIZE` | `500` | Tokens per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `TOP_K` | `5` | Chunks retrieved per query |
| `LLM_MODEL` | `gpt-4o-mini` | OpenAI model |
