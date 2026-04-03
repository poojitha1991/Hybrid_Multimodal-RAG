# Local Multi-Modal RAG Research Assistant

End-to-end, single-screen Streamlit app that ingests local PDFs (text, tables, images), builds a local vector index, and answers questions with citations. Runs fully on your machine — no cloud storage or infra required.

## Features
- Local ingestion from `data/pdfs` or uploads
- Text chunking, basic table extraction, image capture with captions
- ChromaDB persistent vector store
- Sentence-Transformer embeddings (configurable)
- LLM abstraction for Ollama (default) or OpenAI API
- Streamlit chat UI with references (text/table/image)

## Quickstart
1) **Python 3.10+** recommended.  
2) Create venv and install:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```
3) Place PDFs in `data/pdfs` or upload via UI.
4) (Optional) Set env vars in `.env`:
   - `LLM_MODE=ollama | openai | groq`
   - Ollama: `OLLAMA_MODEL=llama3.2:latest`
   - OpenAI: `OPENAI_API_KEY=...`, `OPENAI_MODEL=gpt-4o-mini`
   - Groq: `GROQ_API_KEY=...`, `GROQ_MODEL=llama3-8b-8192`
   - `EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2`
5) Run the app:
   ```bash
   streamlit run app.py
   ```

## Project Layout
- `app.py` — Streamlit UI + chat loop
- `rag/config.py` — paths and tunables
- `rag/ingestion.py` — PDF parsing (text/tables/images) + indexing
- `rag/vectorstore.py` — Chroma wrapper
- `rag/retrieval.py` — search + prompt building + answer orchestration
- `rag/llm.py` — Ollama/OpenAI chat client
- `rag/utils.py` — chunking + shared types
- `data/` — local storage (PDFs, images, Chroma DB)

## Notes
- Table extraction uses `pdfplumber`; quality depends on document structure. For tougher tables consider adding `camelot`/`tabula` (requires Ghostscript/Java).
- Image captions are lightweight (“Image from file page X”) to keep dependencies minimal; you can plug in a vision model to improve them.
- For reproducible research, add your evaluation notebook comparing text-only vs multimodal contexts (see report guidance in prompt).

## Basic Workflow
1) Upload or drop PDFs into `data/pdfs`.
2) Click “Ingest PDFs” to parse and index.
3) Ask questions in the chat. Toggle Deep Scan for broader retrieval.
4) Review references shown under each answer; tables render as data frames, images show inline.


