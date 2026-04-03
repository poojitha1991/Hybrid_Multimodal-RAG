from pathlib import Path
from typing import List
from gtts import gTTS

from . import config


SUMMARY_PROMPT = """You are a document summarizer. Create a clear, concise audio-friendly summary of the document.
Focus on: main topics, key points, important findings. Keep it under 200 words."""


def generate_summary(pdf_path: Path, store, llm) -> str:
    """Generate text summary of PDF."""
    pdf_name = pdf_path.stem
    results = store.collection.get(where={"source": str(pdf_path)})
    
    if not results or not results.get("documents"):
        return f"No content found for {pdf_name}"
    
    docs = results["documents"][:10]
    context = "\n\n".join(docs)
    
    messages = [
        {"role": "system", "content": SUMMARY_PROMPT},
        {"role": "user", "content": f"Summarize this document:\n\n{context}"}
    ]
    
    summary = llm.chat(messages)
    return f"Summary of {pdf_name}. {summary}"


def text_to_speech(text: str, output_path: Path) -> Path:
    """Convert text to audio file."""
    tts = gTTS(text=text, lang='en', slow=False)
    tts.save(str(output_path))
    return output_path


def generate_audio_summary(pdf_path: Path, store, llm) -> Path:
    """Generate audio summary of PDF."""
    config.ensure_directories()
    audio_dir = config.DATA_DIR / "audio"
    audio_dir.mkdir(exist_ok=True)
    
    summary_text = generate_summary(pdf_path, store, llm)
    audio_path = audio_dir / f"{pdf_path.stem}_summary.mp3"
    
    return text_to_speech(summary_text, audio_path)
