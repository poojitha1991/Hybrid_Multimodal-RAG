import os
from pathlib import Path
from typing import List

import streamlit as st

from rag import config
from rag.config import PDF_DIR, ensure_directories
from rag.ingestion import ingest_pdfs
from rag.llm import LLMClient
from rag.retrieval import answer_question
from rag.vectorstore import LocalVectorStore
from rag.audio_summary import generate_audio_summary


def load_existing_pdfs() -> List[Path]:
    ensure_directories()
    return sorted(PDF_DIR.glob("*.pdf"))


def save_uploaded_files(uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile]) -> List[Path]:
    saved_paths: List[Path] = []
    for file in uploaded_files:
        target = PDF_DIR / file.name
        with open(target, "wb") as f:
            f.write(file.getbuffer())
        saved_paths.append(target)
    return saved_paths


def render_sidebar(store: LocalVectorStore, llm: LLMClient) -> dict:
    st.sidebar.header("Documents")
    uploaded = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    deep_scan = st.sidebar.toggle("Deep scan (more context)", value=False)
    if st.sidebar.button("Ingest PDFs"):
        new_paths = save_uploaded_files(uploaded) if uploaded else []
        all_paths = load_existing_pdfs()
        with st.spinner("Indexing PDFs..."):
            stats = ingest_pdfs(all_paths, store)
        st.sidebar.success(f"Ingested {stats['chunks']} chunks from {stats['files']} files.")
    existing = load_existing_pdfs()
    if existing:
        st.sidebar.write("Indexed files:")
        for p in existing:
            st.sidebar.caption(f"• {p.name}")
        
        st.sidebar.divider()
        st.sidebar.subheader("Audio Summary")
        selected_pdf = st.sidebar.selectbox("Select PDF", existing, format_func=lambda x: x.name)
        if st.sidebar.button("Generate Audio Summary"):
            with st.spinner("Generating audio..."):
                try:
                    audio_path = generate_audio_summary(selected_pdf, store, llm)
                    st.sidebar.success("Audio generated!")
                    st.sidebar.audio(str(audio_path))
                except Exception as e:
                    st.sidebar.error(f"Error: {e}")
    else:
        st.sidebar.info("Add PDFs to begin.")
    return {"deep_scan": deep_scan}


def init_clients():
    ensure_directories()
    store = LocalVectorStore()
    llm = LLMClient()
    return store, llm


def render_chat(store: LocalVectorStore, llm: LLMClient, deep_scan: bool):
    st.header("Local Multi-Modal RAG Assistant")
    st.write("Ask about the ingested PDFs. Answers stay grounded in retrieved context.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question about your PDFs"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, references = answer_question(prompt, store, llm, deep_scan=deep_scan)
            st.markdown(answer)
            if references:
                with st.expander("References"):
                    for ref in references:
                        st.markdown(f"- **{ref['source']}**, page {ref['page']} ({ref['modality']})")
                        if ref.get("table"):
                            st.dataframe(ref["table"])
                        if ref.get("image_path") and Path(ref["image_path"]).exists():
                            st.image(ref["image_path"], caption=ref.get("caption"))
        st.session_state.messages.append({"role": "assistant", "content": answer})


def main():
    st.set_page_config(page_title="Local Multi-Modal RAG", layout="wide")
    store, llm = init_clients()
    sidebar_state = render_sidebar(store, llm)
    render_chat(store, llm, deep_scan=sidebar_state["deep_scan"])


if __name__ == "__main__":
    main()


