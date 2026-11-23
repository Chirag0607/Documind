"""Vector store factory: FAISS (in-memory) or ChromaDB (persistent), chosen by VECTOR_STORE."""
import os
import shutil

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

import config

_embeddings = None
_store = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL, api_key=config.OPENAI_API_KEY)
    return _embeddings


def _chroma():
    from langchain_chroma import Chroma

    return Chroma(
        collection_name="documind",
        embedding_function=get_embeddings(),
        persist_directory=config.CHROMA_PERSIST_DIR,
    )


def load_existing():
    """On startup: re-open a persisted Chroma store so answers survive restarts."""
    global _store
    if config.VECTOR_STORE == "chroma" and os.path.isdir(config.CHROMA_PERSIST_DIR):
        _store = _chroma()
    return _store


def get_store():
    return _store


def add_documents(chunks):
    global _store
    if config.VECTOR_STORE == "chroma":
        if _store is None:
            _store = _chroma()
        _store.add_documents(chunks)
    else:
        if _store is None:
            _store = FAISS.from_documents(chunks, get_embeddings())
        else:
            _store.add_documents(chunks)
    return _store


def reset():
    """Clear everything in the store."""
    global _store
    if config.VECTOR_STORE == "chroma":
        if _store is not None:
            try:
                _store.delete_collection()
            except Exception:
                pass
        shutil.rmtree(config.CHROMA_PERSIST_DIR, ignore_errors=True)
    _store = None
