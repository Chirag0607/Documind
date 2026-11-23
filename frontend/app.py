"""Streamlit UI for DocuMind: upload documents, chat, see sources."""
import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="DocuMind", page_icon="📄", layout="wide")
st.title("📄 DocuMind")
st.caption("Ask questions about your documents. Answers come only from what you upload, with sources.")


def api(method: str, path: str, **kwargs):
    try:
        r = requests.request(method, f"{BACKEND_URL}{path}", timeout=120, **kwargs)
        if r.status_code >= 400:
            is_json = r.headers.get("content-type", "").startswith("application/json")
            detail = r.json().get("detail", r.text) if is_json else r.text
            st.error(f"Error {r.status_code}: {detail}")
            return None
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot reach the backend at {BACKEND_URL}. Is it running?")
        return None


def show_sources(sources):
    with st.expander("Sources"):
        for s in sources:
            page = f", page {s['page']}" if s.get("page") else ""
            st.markdown(f"**{s['source']}{page}**  \n_{s['snippet']}..._")


if "session_id" not in st.session_state:
    data = api("POST", "/session/new")
    st.session_state.session_id = data["session_id"] if data else "default"
    st.session_state.messages = []
    st.session_state.files = []

with st.sidebar:
    st.header("Documents")
    uploads = st.file_uploader("Upload PDF, DOCX or TXT", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    if st.button("Process documents", disabled=not uploads, use_container_width=True):
        for f in uploads:
            if f.name in st.session_state.files:
                continue
            with st.spinner(f"Processing {f.name}..."):
                res = api("POST", "/upload", files={"file": (f.name, f.getvalue())})
            if res:
                st.session_state.files.append(f.name)
                st.success(f"{res['filename']}: {res['chunks']} chunks indexed")
    if st.session_state.files:
        st.write("**Indexed:**")
        for name in st.session_state.files:
            st.write(f"- {name}")
    st.divider()
    if st.button("New conversation", use_container_width=True):
        data = api("POST", "/session/new")
        if data:
            st.session_state.session_id = data["session_id"]
            st.session_state.messages = []
            st.rerun()
    if st.button("Reset everything", use_container_width=True):
        if api("DELETE", "/reset"):
            st.session_state.messages = []
            st.session_state.files = []
            st.rerun()

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            show_sources(m["sources"])

if question := st.chat_input("Ask a question about your documents"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            res = api("POST", "/query", json={"question": question, "session_id": st.session_state.session_id})
        if res:
            st.markdown(res["answer"])
            if res["sources"]:
                show_sources(res["sources"])
            st.session_state.messages.append({"role": "assistant", "content": res["answer"], "sources": res["sources"]})
