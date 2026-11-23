from fastapi.testclient import TestClient
from langchain_core.documents import Document

from backend import ingest
from backend.main import app


def test_health():
    r = TestClient(app).get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_chunks_overlap_and_keep_source():
    text = " ".join(f"word{i}" for i in range(3000))
    docs = [Document(page_content=text, metadata={"source": "a.txt"})]
    chunks = ingest.split_documents(docs)
    assert len(chunks) > 1
    assert all(c.metadata["source"] == "a.txt" for c in chunks)


def test_unsupported_type_rejected():
    r = TestClient(app).post("/upload", files={"file": ("x.exe", b"abc")})
    assert r.status_code == 400
