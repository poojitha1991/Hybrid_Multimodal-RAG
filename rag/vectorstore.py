from typing import List

import chromadb
from chromadb.utils import embedding_functions

from . import config
from .utils import DocumentChunk


class LocalVectorStore:
    """Wrapper around Chroma persistent client."""

    def __init__(self):
        config.ensure_directories()
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config.EMBED_MODEL
        )
        self.client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        self.collection = self.client.get_or_create_collection(
            name=config.COLLECTION_NAME, embedding_function=embed_fn
        )

    def add_chunks(self, chunks: List[DocumentChunk]) -> int:
        if not chunks:
            return 0
        self.collection.upsert(
            ids=[c.id for c in chunks],
            documents=[c.content for c in chunks],
            metadatas=[c.metadata for c in chunks],
        )
        return len(chunks)

    def query(self, query_text: str, top_k: int):
        return self.collection.query(
            query_texts=[query_text],
            n_results=top_k,
            include=["metadatas", "documents", "distances"],
        )


