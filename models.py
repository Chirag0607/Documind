from pydantic import BaseModel
from typing import List, Optional


class QueryRequest(BaseModel):
    question: str
    session_id: str = "default"


class SourceDocument(BaseModel):
    source: str
    page: Optional[int] = None
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
    session_id: str


class UploadResponse(BaseModel):
    filename: str
    pages: int
    chunks: int
    vector_store: str
    message: str


class ResetResponse(BaseModel):
    message: str
