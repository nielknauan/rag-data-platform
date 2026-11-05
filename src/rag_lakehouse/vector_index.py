"""Indexação vetorial com LanceDB (formato colunar Lance, embutido, sem servidor).

O índice é um derivado da Gold: pode ser reconstruído a qualquer momento a partir
das tabelas Delta. A sincronização é incremental (merge_insert) e remove vetores
de chunks que saíram da Gold.
"""
from __future__ import annotations

import re

import lancedb
import pandas as pd

ANN_MIN_ROWS = 5_000  # abaixo disso, busca exata (flat) é mais rápida e 100% precisa


def table_name(model_name: str) -> str:
    return "chunks__" + re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")


def sync_index(df: pd.DataFrame, db_path: str, model_name: str) -> dict:
    """df: chunk_id, source, section, text, vector."""
    db = lancedb.connect(db_path)
    name = table_name(model_name)
    if name not in db.list_tables().tables:
        tbl = db.create_table(name, data=df)
    else:
        tbl = db.open_table(name)
        (
            tbl.merge_insert("chunk_id")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .when_not_matched_by_source_delete()
            .execute(df)
        )
    # índice full-text (BM25) para a busca híbrida: acha termos exatos ("encarregado", "portabilidade")
    tbl.create_fts_index("text", replace=True, language="Portuguese", stem=True,
                         remove_stop_words=True, ascii_folding=True)
    rows = tbl.count_rows()
    if rows >= ANN_MIN_ROWS:
        tbl.create_index(metric="cosine", vector_column_name="vector", replace=True)
    return {"table": name, "rows": rows, "ann_index": rows >= ANN_MIN_ROWS}


COLS = ["chunk_id", "source", "section", "text"]
RRF_K = 60  # constante padrão do Reciprocal Rank Fusion


def _vector(tbl, query_vector, n, where):
    q = tbl.search(query_vector).metric("cosine").limit(n)
    if where:
        q = q.where(where, prefilter=True)
    return q.to_pandas()


def _fulltext(tbl, query_text, n, where):
    q = tbl.search(query_text, query_type="fts").limit(n)
    if where:
        q = q.where(where, prefilter=True)
    try:
        return q.to_pandas()
    except Exception:  # sem índice FTS ou consulta sem termos válidos
        return pd.DataFrame(columns=COLS)


def search(
    db_path: str,
    model_name: str,
    query_vector: list[float],
    k: int = 5,
    where: str | None = None,
    query_text: str | None = None,
) -> pd.DataFrame:
    """Busca vetorial; com `query_text`, híbrida (vetor + BM25) fundida por RRF.

    RRF soma 1/(60 + posição) de cada lista: não precisa normalizar escalas de
    score diferentes (distância cosseno vs. BM25) e premia quem aparece bem nas duas.
    """
    tbl = lancedb.connect(db_path).open_table(table_name(model_name))
    vec = _vector(tbl, query_vector, k * 4 if query_text else k, where)
    if not query_text:
        return vec[COLS + ["_distance"]]

    fts = _fulltext(tbl, query_text, k * 4, where)
    scores: dict[str, float] = {}
    for ranked in (vec["chunk_id"].tolist(), fts["chunk_id"].tolist()):
        for pos, cid in enumerate(ranked, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + pos)

    rows = pd.concat([vec[COLS], fts[COLS]]).drop_duplicates("chunk_id").set_index("chunk_id")
    top = sorted(scores, key=scores.get, reverse=True)[:k]
    out = rows.loc[top].reset_index()
    out["_score"] = [round(scores[c], 5) for c in top]
    return out
