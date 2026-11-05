"""Quality gates da Gold. Se algum falhar, o pipeline para antes de publicar o índice."""
from __future__ import annotations

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from .config import MAX_CHUNK_TOKENS, Paths
from .embeddings import get_embedder
from .text import has_pii


class DataQualityError(Exception):
    pass


@F.udf(T.BooleanType())
def _has_pii(text: str) -> bool:
    return has_pii(text or "")


def run_checks(spark: SparkSession, paths: Paths) -> dict:
    chunks = spark.read.format("delta").load(paths.chunks)
    emb = spark.read.format("delta").load(paths.embeddings).filter(F.col("model_name") == get_embedder().name)
    joined = chunks.join(emb, "chunk_id", "left")

    results = {
        "chunks_sem_pii": chunks.filter(_has_pii("text")).count() == 0,
        "chunks_nao_vazios": chunks.filter(F.length(F.trim("text")) == 0).count() == 0,
        "chunks_tamanho_ok": chunks.filter(F.col("n_tokens") > MAX_CHUNK_TOKENS).count() == 0,
        "chunk_id_unico": chunks.count() == chunks.select("chunk_id").distinct().count(),
        "todo_chunk_tem_vetor": joined.filter(F.col("vector").isNull()).count() == 0,
        "dimensao_unica": emb.select("dim").distinct().count() <= 1,
    }
    failed = [k for k, ok in results.items() if not ok]
    if failed:
        raise DataQualityError(f"Quality gates falharam: {failed}")
    return results
