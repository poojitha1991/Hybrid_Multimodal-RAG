import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
IMAGE_DIR = DATA_DIR / "images"
CHROMA_DIR = DATA_DIR / "chroma"

COLLECTION_NAME = "multimodal_docs"
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# LLM settings
LLM_MODE = os.getenv("LLM_MODE", "groq")  # "groq" | "ollama" | "openai"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Chunking / retrieval
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K", "6"))
DEEP_TOP_K = int(os.getenv("DEEP_TOP_K", "12"))


def ensure_directories() -> None:
    """Create data directories if missing."""
    for path in (DATA_DIR, PDF_DIR, IMAGE_DIR, CHROMA_DIR):
        path.mkdir(parents=True, exist_ok=True)


