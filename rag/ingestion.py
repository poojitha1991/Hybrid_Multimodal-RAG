from pathlib import Path
from typing import Dict, List

import pandas as pd
import pdfplumber
import fitz  # PyMuPDF

from . import config
from .utils import DocumentChunk, chunk_text, make_id


def extract_text_and_tables(pdf_path: Path) -> List[DocumentChunk]:
    chunks: List[DocumentChunk] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            for idx, chunk in enumerate(
                chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
            ):
                chunks.append(
                    DocumentChunk(
                        id=make_id(f"{pdf_path.stem}-p{page_idx}-t{idx}"),
                        content=chunk,
                        metadata={
                            "doc_id": pdf_path.stem,
                            "source": str(pdf_path),
                            "page": page_idx,
                            "modality": "text",
                            "chunk_index": idx,
                        },
                    )
                )
            tables = page.extract_tables() or []
            for t_idx, table in enumerate(tables):
                df = pd.DataFrame(table)
                caption = f"Table from {pdf_path.name} page {page_idx}"
                chunks.append(
                    DocumentChunk(
                        id=make_id(f"{pdf_path.stem}-p{page_idx}-table{t_idx}"),
                        content=f"{caption}\n{df.to_markdown(index=False)}",
                        metadata={
                            "doc_id": pdf_path.stem,
                            "source": str(pdf_path),
                            "page": page_idx,
                            "modality": "table",
                            "caption": caption,
                            "table": df.to_dict(orient="records"),
                        },
                    )
                )
    return chunks


def extract_images(pdf_path: Path) -> List[DocumentChunk]:
    chunks: List[DocumentChunk] = []
    doc = fitz.open(pdf_path)
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        for img_index, img in enumerate(page.get_images(full=True), start=1):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            if pix.n - pix.alpha > 3:  # convert to RGB if CMYK/others
                pix = fitz.Pixmap(fitz.csRGB, pix)
            image_path = config.IMAGE_DIR / f"{pdf_path.stem}_p{page_idx+1}_img{img_index}.png"
            pix.save(image_path)
            caption = f"Image from {pdf_path.name} page {page_idx+1}"
            chunks.append(
                DocumentChunk(
                    id=make_id(f"{pdf_path.stem}-p{page_idx+1}-img{img_index}"),
                    content=caption,
                    metadata={
                        "doc_id": pdf_path.stem,
                        "source": str(pdf_path),
                        "page": page_idx + 1,
                        "modality": "image",
                        "caption": caption,
                        "image_path": str(image_path),
                    },
                )
            )
    return chunks


def ingest_pdfs(pdf_paths: List[Path], store) -> Dict[str, int]:
    """Parse PDFs into chunks and add to vector store."""
    total_chunks = 0
    seen_files = 0
    config.ensure_directories()
    for path in pdf_paths:
        if not path.exists() or path.suffix.lower() != ".pdf":
            continue
        text_chunks = extract_text_and_tables(path)
        image_chunks = extract_images(path)
        added = store.add_chunks(text_chunks + image_chunks)
        total_chunks += added
        seen_files += 1
    return {"files": seen_files, "chunks": total_chunks}


