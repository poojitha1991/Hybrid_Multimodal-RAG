from dataclasses import dataclass, field
from typing import Dict, List
import uuid


@dataclass
class DocumentChunk:
    id: str
    content: str
    metadata: Dict


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Simple whitespace chunker with overlap."""
    words = text.split()
    if not words:
        return []
    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        start += max(chunk_size - overlap, 1)
    return chunks


def make_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


