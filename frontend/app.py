import os

import requests
import streamlit as st

# Backend URL: set API_URL when running in Docker, defaults to local
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Hybrid RAG + Knowledge Graph", page_icon="🕸️", layout="wide")
st.title("Hybrid RAG + Knowledge Graph")
st.caption("Vector search (Chroma) + knowledge graph (Neo4j) over your PDF")

if "messages" not in st.session_state:
    st.session_state.messages = []


def error_text(resp: requests.Response) -> str:
    try:
        return str(resp.json().get("detail", resp.text))
    except Exception:
        return resp.text


def show_assistant(msg: dict):
    """Show the answer, plus what the vector search and the graph contributed."""
    st.write(msg["content"])
    with st.expander(f"Graph facts used ({len(msg['graph_facts'])})"):
        st.write("**Entities matched:**", ", ".join(msg["entities"]) or "none")
        for fact in msg["graph_facts"]:
            st.code(fact, language=None)
    st.caption("Vector search used pages: " + ", ".join(msg["pages"]))


# ---------------- Sidebar: PDF upload ----------------
with st.sidebar:
    st.header("1. Upload a PDF")
    uploaded = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded and st.button("Ingest PDF", type="primary"):
        with st.spinner("Building the vector index and the knowledge graph (30-90 seconds)..."):
            try:
                resp = requests.post(
                    f"{API_URL}/ingest",
                    files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                    timeout=600,
                )
                if resp.ok:
                    data = resp.json()
                    st.success(f"Ingested '{data['doc_id']}'")
                    st.write(f"Pages: **{data['pages']}**")
                    st.write(f"Vector chunks: **{data['vector_chunks']}**")
                    st.write(f"Graph relationships: **{data['graph']['relationships']}**")
                else:
                    st.error(error_text(resp))
            except requests.exceptions.ConnectionError:
                st.error(f"Cannot reach the backend at {API_URL}. Is it running?")

    st.divider()
    st.header("2. Ask questions")
    st.write("Type in the box at the bottom of the page.")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---------------- Chat history ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            show_assistant(msg)
        else:
            st.write(msg["content"])

# ---------------- New question ----------------
question = st.chat_input("Ask a question about your PDF")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching vectors and the graph..."):
            try:
                resp = requests.post(f"{API_URL}/ask", json={"question": question}, timeout=120)
                if resp.ok:
                    data = resp.json()
                    msg = {
                        "role": "assistant",
                        "content": data["answer"],
                        "entities": data["entities"],
                        "graph_facts": data["graph_facts"],
                        "pages": [str(p) for p in data["pages"]],
                    }
                    st.session_state.messages.append(msg)
                    show_assistant(msg)
                else:
                    st.error(error_text(resp))
            except requests.exceptions.ConnectionError:
                st.error(f"Cannot reach the backend at {API_URL}. Is it running?")