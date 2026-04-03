import os
from typing import List

import requests
from . import config

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    from groq import Groq
except Exception:
    Groq = None


class LLMClient:
    """Simple abstraction over Ollama, OpenAI, or Groq chat APIs."""

    def __init__(self):
        self.mode = config.LLM_MODE.lower()
        self.ollama_model = config.OLLAMA_MODEL
        self.openai_model = config.OPENAI_MODEL
        self.groq_model = config.GROQ_MODEL
        self.openai_key = config.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
        self.groq_key = config.GROQ_API_KEY or os.getenv("GROQ_API_KEY")

        if self.mode == "openai":
            if not self.openai_key:
                raise ValueError("OPENAI_API_KEY is required for OpenAI mode.")
            if OpenAI is None:
                raise ImportError("openai package missing; install to use OpenAI mode.")
        if self.mode == "groq":
            if not self.groq_key:
                raise ValueError("GROQ_API_KEY is required for Groq mode.")
            if Groq is None:
                raise ImportError("groq package missing; install to use Groq mode.")

    def chat(self, messages: List[dict]) -> str:
        if self.mode == "openai":
            return self._chat_openai(messages)
        if self.mode == "groq":
            return self._chat_groq(messages)
        return self._chat_ollama(messages)

    def _chat_openai(self, messages: List[dict]) -> str:
        client = OpenAI(api_key=self.openai_key)
        resp = client.chat.completions.create(
            model=self.openai_model, messages=messages, temperature=0.2
        )
        return resp.choices[0].message.content

    def _chat_groq(self, messages: List[dict]) -> str:
        client = Groq(api_key=self.groq_key)
        resp = client.chat.completions.create(
            model=self.groq_model, messages=messages, temperature=0.2
        )
        return resp.choices[0].message.content

    def _chat_ollama(self, messages: List[dict]) -> str:
        payload = {"model": self.ollama_model, "messages": messages, "stream": False}
        resp = requests.post("http://localhost:11434/api/chat", json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        if "message" in data:
            return data["message"].get("content", "")
        return data.get("response", "")


