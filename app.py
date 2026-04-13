import uuid
import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="DocuMind", page_icon="📚", layout="wide")
st.title("📚 DocuMind")
st.caption("RAG-based Knowledge Assistant — Ask anything from your documents")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# ---- Sidebar ----
with st.sidebar:
    st.header("📂 Upload Documents")
    files = st.file_uploader(
        "Supports PDF, DOCX, TXT",
        type=["pdf", "docx", "doc", "txt"],
        accept_multiple_files=True,
    )

    if files:
        for f in files:
            if f.name not in st.session_state.uploaded_files:
                with st.spinner(f"Ingesting {f.name}…"):
                    resp = requests.post(
                        f"{API_URL}/upload",
                        files={"file": (f.name, f.getvalue(), f.type)},
                        timeout=60,
                    )
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(f" {f.name} — {data['chunks']} chunks")
                    st.session_state.uploaded_files.append(f.name)
                else:
                    st.error(f" {resp.json().get('detail', 'Error')}")

    if st.session_state.uploaded_files:
        st.divider()
        st.subheader("Ingested")
        for name in st.session_state.uploaded_files:
            st.markdown(f" {name}")

    st.divider()
    if st.button("🗑️ Clear Everything", use_container_width=True):
        requests.delete(f"{API_URL}/reset?session_id={st.session_state.session_id}", timeout=10)
        st.session_state.messages = []
        st.session_state.uploaded_files = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    try:
        h = requests.get(f"{API_URL}/health", timeout=2).json()
        st.caption(f"Store: `{h.get('vector_store')}`  |  Session: `{st.session_state.session_id[:8]}…`")
    except Exception:
        st.caption(" Backend unreachable")

# ---- Chat ----
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 Sources"):
                for s in msg["sources"]:
                    pg = f" · page {s['page']}" if s.get("page") is not None else ""
                    st.markdown(f"**{s['source']}{pg}**")
                    st.caption(s["snippet"])

if prompt := st.chat_input("Ask a question about your documents…"):
    if not st.session_state.uploaded_files:
        st.warning("Upload at least one document first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            resp = requests.post(
                f"{API_URL}/query",
                json={"question": prompt, "session_id": st.session_state.session_id},
                timeout=60,
            )
        if resp.status_code == 200:
            data = resp.json()
            st.markdown(data["answer"])
            if data.get("sources"):
                with st.expander("📎 Sources"):
                    for s in data["sources"]:
                        pg = f" · page {s['page']}" if s.get("page") is not None else ""
                        st.markdown(f"**{s['source']}{pg}**")
                        st.caption(s["snippet"])
            st.session_state.messages.append({
                "role": "assistant",
                "content": data["answer"],
                "sources": data.get("sources", []),
            })
        else:
            err = resp.json().get("detail", "Query failed")
            st.error(err)
            st.session_state.messages.append({"role": "assistant", "content": f"Error: {err}"})
