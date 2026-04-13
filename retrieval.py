from typing import Dict, Any
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_classic.prompts import PromptTemplate

import config
from backend.ingestion import get_vector_store

llm = ChatOpenAI(
    openai_api_key=config.OPENAI_API_KEY,
    model=config.LLM_MODEL,
    temperature=config.TEMPERATURE,
    max_tokens=config.MAX_TOKENS,
)

_QA_PROMPT = PromptTemplate(
    template=(
        "You are a helpful document assistant. Answer the question ONLY from the provided context.\n"
        "If the answer is not in the context, respond: 'I don't have that information in the uploaded documents.'\n"
        "Always cite the source document name.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    ),
    input_variables=["context", "question"],
)

_chains: Dict[str, ConversationalRetrievalChain] = {}


def get_chain(session_id: str) -> ConversationalRetrievalChain:
    if session_id not in _chains:
        store = get_vector_store()
        if store is None:
            raise ValueError("No documents ingested yet. Please upload a document first.")

        retriever = store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": config.TOP_K,
                "fetch_k": config.TOP_K * 3,
                "lambda_mult": 0.5,
            },
        )
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
        )
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": _QA_PROMPT},
        )
        _chains[session_id] = chain
    return _chains[session_id]


def query(session_id: str, question: str) -> Dict[str, Any]:
    chain = get_chain(session_id)
    result = chain({"question": question})

    sources = []
    seen: set = set()
    for doc in result.get("source_documents", []):
        src = doc.metadata.get("source", "Unknown")
        if src not in seen:
            sources.append({
                "source": src,
                "page": doc.metadata.get("page"),
                "snippet": doc.page_content[:220] + "...",
            })
            seen.add(src)

    return {"answer": result["answer"], "sources": sources, "session_id": session_id}


def reset_session(session_id: str) -> None:
    _chains.pop(session_id, None)
