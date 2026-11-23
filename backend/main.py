"""FastAPI backend: upload documents, ask questions, manage sessions."""
import logging
import os
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

import config
from backend import chain, ingest, vectorstore

logger = logging.getLogger("documind")
MAX_UPLOAD_MB = 20


@asynccontextmanager
async def lifespan(app: FastAPI):
    if config.OPENAI_API_KEY:
        vectorstore.load_existing()
    yield


app = FastAPI(title="DocuMind API", version="1.0.0", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str
    session_id: str = "default"


def _require_key():
    if not config.OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set. Add it to your .env file.")


@app.get("/health")
def health():
    return {"status": "ok", "vector_store": config.VECTOR_STORE, "model": config.LLM_MODEL}


@app.post("/upload")
def upload(file: UploadFile = File(...)):
    """Receive a file, extract text, chunk it, embed it, store it."""
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ingest.SUPPORTED:
        raise HTTPException(status_code=400, detail="Supported file types: PDF, DOCX, TXT.")
    _require_key()
    data = file.file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large (max {MAX_UPLOAD_MB} MB).")
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        chunks = ingest.ingest_file(path, file.filename)
        vectorstore.add_documents(chunks)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=502, detail=f"Ingestion failed: {e}")
    finally:
        os.remove(path)
    return {"filename": file.filename, "chunks": len(chunks)}


@app.post("/query")
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question is empty.")
    _require_key()
    if vectorstore.get_store() is None:
        return {"answer": "Please upload a document first.", "sources": [], "session_id": req.session_id}
    try:
        return chain.ask(req.question, req.session_id)
    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(status_code=502, detail=f"Query failed: {e}")


@app.delete("/reset")
def reset():
    vectorstore.reset()
    chain.clear_sessions()
    return {"status": "cleared"}


@app.post("/session/new")
def new_session():
    return {"session_id": chain.new_session()}
