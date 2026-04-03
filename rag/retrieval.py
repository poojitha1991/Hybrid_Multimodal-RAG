from typing import List, Tuple, Dict
from pathlib import Path
from collections import defaultdict
import re

from . import config


SYSTEM_PROMPT = """You are a precise research assistant. Answer strictly using the provided context from the uploaded documents (text, tables, images).
If the context does not contain the answer, say exactly: "I don't find this information in the uploaded documents."."""


def _expand_query(question: str) -> List[str]:
    """Generate query variations for 'what is' questions."""
    queries = [question]
    lower_q = question.lower().strip()
    if lower_q.startswith(("what is", "what are", "define")):
        base = re.sub(r"^(what is|what are|define)\s+", "", lower_q, flags=re.IGNORECASE)
        queries.append(f"definition of {base}")
        queries.append(f"explain {base}")
    return queries


def _keyword_score(query: str, text: str) -> float:
    """Calculate term overlap score."""
    q_terms = set(re.findall(r"\w+", query.lower()))
    t_terms = set(re.findall(r"\w+", text.lower()))
    if not q_terms:
        return 0.0
    overlap = len(q_terms & t_terms)
    return overlap / len(q_terms)


def _hybrid_retrieve(question: str, store, top_k: int) -> Dict:
    """Hybrid retrieval with query expansion and scoring."""
    queries = _expand_query(question)
    all_results = defaultdict(lambda: {"score": 0.0, "doc": "", "meta": {}})
    
    for query in queries:
        results = store.query(query, top_k=top_k * 2)
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        
        for doc, meta, dist in zip(docs, metas, distances):
            chunk_id = f"{meta.get('source', '')}_{meta.get('page', '')}_{doc[:50]}"
            semantic_score = 1 / (1 + dist)
            keyword_score = _keyword_score(question, doc)
            hybrid_score = 0.3 * keyword_score + 0.7 * semantic_score
            
            if meta.get("modality") == "text":
                hybrid_score *= 1.2
            
            if hybrid_score > all_results[chunk_id]["score"]:
                all_results[chunk_id] = {"score": hybrid_score, "doc": doc, "meta": meta}
    
    ranked = sorted(all_results.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    
    return {
        "documents": [[r["doc"] for r in ranked]],
        "metadatas": [[r["meta"] for r in ranked]],
    }


def _format_context(results) -> Tuple[str, List[dict]]:
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    lines: List[str] = []
    references: List[dict] = []
    for doc, meta in zip(docs, metas):
        source = Path(meta.get("source", "")).name
        modality = meta.get("modality", "text")
        page = meta.get("page")
        cite = f"[{modality.upper()} from {source}, page {page}]"
        lines.append(f"{cite}\n{doc}")
        references.append(
            {
                "source": source,
                "page": page,
                "modality": modality,
                "caption": meta.get("caption"),
                "table": meta.get("table"),
                "image_path": meta.get("image_path"),
            }
        )
    context_block = "\n\n".join(lines)
    return context_block, references


def build_messages(question: str, context: str) -> List[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]


def answer_question(
    question: str, store, llm, deep_scan: bool = False
) -> Tuple[str, List[dict]]:
    top_k = config.DEEP_TOP_K if deep_scan else config.TOP_K
    search_results = _hybrid_retrieve(question, store, top_k)
    context, references = _format_context(search_results)
    if not context.strip():
        return "I don't find this information in the uploaded documents.", []
    messages = build_messages(question, context)
    answer = llm.chat(messages)
    return answer, references


