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
    rows = tbl.count_rows()
    if rows >= ANN_MIN_ROWS:
        tbl.create_index(metric="cosine", vector_column_name="vector", replace=True)
    return {"table": name, "rows": rows, "ann_index": rows >= ANN_MIN_ROWS}


def search(db_path: str, model_name: str, query_vector: list[float], k: int = 5, where: str | None = None) -> pd.DataFrame:
    tbl = lancedb.connect(db_path).open_table(table_name(model_name))
    q = tbl.search(query_vector).metric("cosine").limit(k)
    if where:
        q = q.where(where, prefilter=True)
    return q.to_pandas()[["chunk_id", "source", "section", "text", "_distance"]]
