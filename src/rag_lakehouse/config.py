"""Caminhos e parâmetros do pipeline."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "1500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
MAX_CHUNK_TOKENS = 600  # limite de qualidade: chunk maior que isso degrada o retrieval


@dataclass(frozen=True)
class Paths:
    root: Path

    def _p(self, *parts: str) -> str:
        return str(self.root.joinpath(*parts))

    @property
    def landing(self) -> str:
        return self._p("landing", "docs")

    @property
    def bronze(self) -> str:
        return self._p("bronze", "documents")

    @property
    def silver(self) -> str:
        return self._p("silver", "documents")

    @property
    def chunks(self) -> str:
        return self._p("gold", "chunks")

    @property
    def embeddings(self) -> str:
        return self._p("gold", "embeddings")

    @property
    def vector_db(self) -> str:
        return self._p("vector", "lancedb")

    def checkpoint(self, name: str) -> str:
        return self._p("_checkpoints", name)


def get_paths() -> Paths:
    return Paths(Path(os.getenv("LAKEHOUSE_ROOT", "lakehouse")).resolve())
