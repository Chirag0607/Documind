"""Conversational RAG chain: MMR retrieval + grounded prompt + per-session memory."""
import uuid

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

import config
from backend import vectorstore

NOT_FOUND = "I couldn't find this in the uploaded documents."

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are DocuMind, an assistant that answers questions about the user's documents.\n"
        "Answer ONLY using the context below. Do not use outside knowledge.\n"
        f"If the answer is not in the context, reply exactly: {NOT_FOUND}\n"
        "Be concise and accurate.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    ),
)

_memories = {}  # session_id -> ConversationBufferMemory


def new_session() -> str:
    sid = uuid.uuid4().hex
    _memories[sid] = _make_memory()
    return sid


def clear_sessions():
    _memories.clear()


def _make_memory():
    return ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")


def _get_memory(session_id: str):
    if session_id not in _memories:
        _memories[session_id] = _make_memory()
    return _memories[session_id]


def build_chain(session_id: str, llm=None):
    store = vectorstore.get_store()
    retriever = store.as_retriever(
        search_type="mmr",  # relevance AND diversity
        search_kwargs={"k": config.TOP_K, "fetch_k": config.TOP_K * 4},
    )
    llm = llm or ChatOpenAI(
        model=config.LLM_MODEL,
        temperature=config.TEMPERATURE,
        max_tokens=config.MAX_TOKENS,
        api_key=config.OPENAI_API_KEY,
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=_get_memory(session_id),
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
    )


def ask(question: str, session_id: str, llm=None) -> dict:
    chain = build_chain(session_id, llm=llm)
    result = chain.invoke({"question": question})
    answer = result["answer"].strip()
    sources, seen = [], set()
    if NOT_FOUND not in answer:  # don't cite sources for a "not found" answer
        for doc in result.get("source_documents", []):
            key = (doc.metadata.get("source"), doc.metadata.get("page"))
            if key in seen:
                continue
            seen.add(key)
            page = doc.metadata.get("page")
            sources.append(
                {
                    "source": doc.metadata.get("source", "unknown"),
                    "page": (page + 1) if isinstance(page, int) else None,
                    "snippet": doc.page_content[:200].replace("\n", " "),
                }
            )
    return {"answer": answer, "sources": sources, "session_id": session_id}
