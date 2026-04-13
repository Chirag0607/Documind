import os
import tempfile
from typing import List, Tuple
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.vectorstores import FAISS, Chroma
from langchain.schema import Document

import config

embeddings = OpenAIEmbeddings(
    openai_api_key=config.OPENAI_API_KEY,
    model=config.EMBEDDING_MODEL,
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.CHUNK_SIZE,
    chunk_overlap=config.CHUNK_OVERLAP,
    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
)

_vector_store = None


def _load_document(file_path: str, filename: str) -> List[Document]:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext in [".docx", ".doc"]:
        loader = Docx2txtLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return loader.load()


def ingest_document(file_bytes: bytes, filename: str) -> Tuple[int, int]:
    global _vector_store

    suffix = Path(filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        documents = _load_document(tmp_path, filename)
        for doc in documents:
            doc.metadata["source"] = filename

        chunks = text_splitter.split_documents(documents)

        if config.VECTOR_STORE == "chroma":
            if _vector_store is None:
                _vector_store = Chroma(
                    persist_directory=config.CHROMA_PERSIST_DIR,
                    embedding_function=embeddings,
                )
            _vector_store.add_documents(chunks)
            _vector_store.persist()
        else:
            if _vector_store is None:
                _vector_store = FAISS.from_documents(chunks, embeddings)
            else:
                new_store = FAISS.from_documents(chunks, embeddings)
                _vector_store.merge_from(new_store)

        return len(documents), len(chunks)
    finally:
        os.unlink(tmp_path)


def get_vector_store():
    global _vector_store
    if _vector_store is None and config.VECTOR_STORE == "chroma":
        if os.path.exists(config.CHROMA_PERSIST_DIR):
            _vector_store = Chroma(
                persist_directory=config.CHROMA_PERSIST_DIR,
                embedding_function=embeddings,
            )
    return _vector_store


def reset_vector_store():
    global _vector_store
    _vector_store = None
    if config.VECTOR_STORE == "chroma":
        import shutil
        if os.path.exists(config.CHROMA_PERSIST_DIR):
            shutil.rmtree(config.CHROMA_PERSIST_DIR)
