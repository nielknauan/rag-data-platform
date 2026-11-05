"""Embedders plugáveis.

- `fastembed` (padrão): modelo multilíngue real, roda em CPU via ONNX, sem GPU nem API paga.
- `hash`: determinístico e offline, usado nos testes unitários (não tem semântica).

O nome + versão do modelo fazem parte da chave da tabela de embeddings: trocar de
modelo gera vetores novos sem apagar os antigos (versionamento / rollback).
"""
from __future__ import annotations

import hashlib
import math
import os
import re
from typing import Protocol

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashEmbedder:
    """Bag-of-words com hashing trick, TF sublinear e normalização L2. Barato e reprodutível.

    Ignora tokens curtos (stopwords como "de", "a", "o") para não dominarem o vetor.
    """

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.name = f"hash-bow-{dim}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = [0.0] * self.dim
            tf: dict[str, int] = {}
            for tok in re.findall(r"\w{4,}", t.lower()):
                tf[tok] = tf.get(tok, 0) + 1
            for tok, n in tf.items():
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                v[h % self.dim] += (1.0 if (h >> 8) % 2 else -1.0) * (1 + math.log(n))
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / norm for x in v])
        return out


class FastEmbedder:
    def __init__(self, model: str = DEFAULT_MODEL, batch_size: int = 64):
        from fastembed import TextEmbedding  # import tardio: só carrega se usado

        self._model = TextEmbedding(model_name=model)
        self.name = model
        self.batch_size = batch_size
        self.dim = len(next(iter(self._model.embed(["dim"]))))

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.embed(texts, batch_size=self.batch_size)]


_cache: dict[str, Embedder] = {}


def get_embedder(kind: str | None = None) -> Embedder:
    kind = kind or os.getenv("EMBEDDER", "fastembed")
    if kind not in _cache:
        _cache[kind] = HashEmbedder() if kind == "hash" else FastEmbedder(os.getenv("EMBED_MODEL", DEFAULT_MODEL))
    return _cache[kind]
