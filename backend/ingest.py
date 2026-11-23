"""Ingestion: load PDF / DOCX / TXT files and split them into overlapping chunks."""
from pathlib import Path

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

SUPPORTED = {".pdf", ".docx", ".txt"}


def load_file(path: str, original_name: str):
    """Pick a loader by file extension and tag every page with the original file name."""
    ext = Path(original_name).suffix.lower()
    if ext == ".pdf":
        loader = PyPDFLoader(path)
    elif ext == ".docx":
        loader = Docx2txtLoader(path)
    elif ext == ".txt":
        loader = TextLoader(path, encoding="utf-8", autodetect_encoding=True)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    docs = loader.load()
    for d in docs:
        d.metadata["source"] = original_name  # used later for citations
    return docs


def get_splitter() -> RecursiveCharacterTextSplitter:
    """Token-based splitter (CHUNK_SIZE tokens). Falls back to characters if tiktoken data is unavailable."""
    try:
        return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
        )
    except Exception:
        return RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE * 4, chunk_overlap=config.CHUNK_OVERLAP * 4
        )


def split_documents(docs):
    return get_splitter().split_documents(docs)


def ingest_file(path: str, original_name: str):
    """Load -> check there is text -> split into overlapping chunks."""
    docs = load_file(path, original_name)
    if not any(d.page_content.strip() for d in docs):
        raise ValueError("No extractable text found (is this a scanned PDF?).")
    return split_documents(docs)
