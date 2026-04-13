import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
VECTOR_STORE      = os.getenv("VECTOR_STORE", "faiss")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CHUNK_SIZE        = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP     = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K             = int(os.getenv("TOP_K", "5"))
EMBEDDING_MODEL   = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL         = os.getenv("LLM_MODEL", "gpt-4o-mini")
MAX_TOKENS        = int(os.getenv("MAX_TOKENS", "1000"))
TEMPERATURE       = float(os.getenv("TEMPERATURE", "0.1"))
