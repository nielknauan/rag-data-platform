"""Avaliação do retrieval com um golden set: hit rate@k e MRR.

Roda no CI como gate: se a qualidade da busca cair (troca de modelo, chunking,
limpeza), o build quebra antes de chegar em produção.
"""
from __future__ import annotations

import json
from pathlib import Path


def hit_and_rank(retrieved: list[tuple[str, str]], expected: tuple[str, str]) -> int | None:
    """Posição (1-based) do primeiro chunk que bate (source, section) esperado."""
    for i, item in enumerate(retrieved, start=1):
        if item == expected:
            return i
    return None


def metrics(ranks: list[int | None], k: int) -> dict:
    n = len(ranks)
    hits = sum(1 for r in ranks if r is not None and r <= k)
    mrr = sum(1 / r for r in ranks if r is not None and r <= k) / n if n else 0.0
    return {f"hit_rate@{k}": round(hits / n, 3) if n else 0.0, f"mrr@{k}": round(mrr, 3), "n": n}


def run_eval(golden_path: str, k: int = 5, include_incremental: bool = True) -> dict:
    from .rag import retrieve

    rows = [json.loads(line) for line in Path(golden_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not include_incremental:
        rows = [r for r in rows if not r.get("after_incremental")]
    # Compara as duas estratégias no mesmo golden set; o gate usa a híbrida (padrão)
    out = {}
    for mode in ("vector", "hybrid"):
        ranks, details = [], []
        for row in rows:
            hits = retrieve(row["question"], k=k, hybrid=mode == "hybrid")
            got = list(zip(hits["source"], hits["section"]))
            rank = hit_and_rank(got, (row["source"], row["section"]))
            ranks.append(rank)
            details.append({"question": row["question"], "expected": row["section"], "rank": rank})
        out[mode] = metrics(ranks, k)
    return {**out["hybrid"], "comparison": out, "details": details}
