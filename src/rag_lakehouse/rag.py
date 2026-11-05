"""Retrieval + geração com citação de fonte. O LLM é plugável:

LLM_PROVIDER=none       -> resposta extrativa (trechos recuperados), sem custo, usado no CI
LLM_PROVIDER=ollama     -> modelo local (OLLAMA_MODEL, padrão llama3.1)
LLM_PROVIDER=openai     -> OPENAI_API_KEY / OPENAI_MODEL
LLM_PROVIDER=anthropic  -> ANTHROPIC_API_KEY / ANTHROPIC_MODEL
"""
from __future__ import annotations

import json
import os
import urllib.request

from .config import get_paths
from .embeddings import get_embedder
from .vector_index import search

SYSTEM = (
    "Você responde dúvidas sobre proteção de dados usando APENAS o contexto fornecido. "
    "Cite as fontes no formato [n]. Se o contexto não tiver a resposta, diga que não sabe."
)


def retrieve(question: str, k: int = 5, where: str | None = None, hybrid: bool | None = None):
    hybrid = os.getenv("SEARCH_MODE", "hybrid") == "hybrid" if hybrid is None else hybrid
    emb = get_embedder()
    return search(get_paths().vector_db, emb.name, emb.embed([question])[0], k=k, where=where,
                  query_text=question if hybrid else None)


def _post(url: str, payload: dict, headers: dict) -> dict:
    req = urllib.request.Request(url, json.dumps(payload).encode(), {"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def generate(question: str, context: str) -> str:
    provider = os.getenv("LLM_PROVIDER", "none")
    user = f"Contexto:\n{context}\n\nPergunta: {question}"
    if provider == "ollama":
        r = _post(
            os.getenv("OLLAMA_URL", "http://localhost:11434") + "/api/chat",
            {"model": os.getenv("OLLAMA_MODEL", "llama3.1"), "stream": False,
             "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]},
            {},
        )
        return r["message"]["content"]
    if provider == "openai":
        r = _post(
            "https://api.openai.com/v1/chat/completions",
            {"model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
             "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]},
            {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
        )
        return r["choices"][0]["message"]["content"]
    if provider == "anthropic":
        r = _post(
            "https://api.anthropic.com/v1/messages",
            {"model": os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5"), "max_tokens": 800, "system": SYSTEM,
             "messages": [{"role": "user", "content": user}]},
            {"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"},
        )
        return r["content"][0]["text"]
    return "Trechos mais relevantes:\n\n" + context


def answer(question: str, k: int = 4) -> dict:
    hits = retrieve(question, k=k)
    rows = hits.to_dict("records")
    context = "\n\n".join(f"[{i + 1}] ({r['source']} · {r['section']})\n{r['text']}" for i, r in enumerate(rows))
    return {
        "question": question,
        "answer": generate(question, context),
        "sources": [{"n": i + 1, "source": r["source"], "section": r["section"], "score": round(float(r.get("_score", r.get("_distance", 0))), 4)}
                    for i, r in enumerate(rows)],
    }
