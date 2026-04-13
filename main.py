import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.models import QueryRequest, QueryResponse, UploadResponse, ResetResponse
from backend.ingestion import ingest_document, reset_vector_store
from backend.retrieval import query, reset_session
import config

app = FastAPI(
    title="DocuMind API",
    description="RAG-based Internal Knowledge Assistant",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt"}


@app.get("/health")
def health():
    return {"status": "ok", "vector_store": config.VECTOR_STORE}


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported type '{ext}'. Allowed: {ALLOWED_EXTENSIONS}")

    content = await file.read()
    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(400, "File too large. Max 20 MB.")

    try:
        pages, chunks = ingest_document(content, file.filename)
        return UploadResponse(
            filename=file.filename,
            pages=pages,
            chunks=chunks,
            vector_store=config.VECTOR_STORE,
            message=f"Ingested '{file.filename}': {pages} page(s) → {chunks} chunk(s)",
        )
    except Exception as exc:
        raise HTTPException(500, f"Ingestion failed: {exc}")


@app.post("/query", response_model=QueryResponse)
async def query_documents(req: QueryRequest):
    try:
        result = query(req.session_id, req.question)
        return QueryResponse(**result)
    except ValueError as exc:
        raise HTTPException(404, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"Query failed: {exc}")


@app.delete("/reset", response_model=ResetResponse)
def reset(session_id: str = "default"):
    reset_vector_store()
    reset_session(session_id)
    return ResetResponse(message="Vector store and session cleared.")


@app.post("/session/new")
def new_session():
    return {"session_id": str(uuid.uuid4())}
