"""Gold: dataset de chunks (pronto para ML) e tabela de embeddings incremental.

A tabela `embeddings` funciona como uma feature store de vetores:
chave = (chunk_id, model_name). Um chunk só é embedado uma vez por modelo;
reprocessar o pipeline sem mudança de conteúdo custa zero chamadas ao modelo.
"""
from __future__ import annotations

from collections.abc import Iterator

import pandas as pd
from delta.tables import DeltaTable
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from .config import CHUNK_MAX_CHARS, CHUNK_OVERLAP, Paths
from .embeddings import get_embedder
from .text import chunk_document, estimate_tokens, sha256

_chunk_schema = T.ArrayType(
    T.StructType(
        [
            T.StructField("chunk_index", T.IntegerType()),
            T.StructField("section", T.StringType()),
            T.StructField("text", T.StringType()),
        ]
    )
)


@F.udf(_chunk_schema)
def _chunk(content: str):
    return chunk_document(content or "", CHUNK_MAX_CHARS, CHUNK_OVERLAP)


@F.udf(T.StringType())
def _chunk_id(doc_id: str, section: str, text: str) -> str:
    return sha256(doc_id, section, text)


@F.udf(T.IntegerType())
def _tokens(text: str) -> int:
    return estimate_tokens(text or "")


def build_chunks(silver: DataFrame) -> DataFrame:
    return (
        silver.select("doc_id", "file_name", "content_hash", F.explode(_chunk("content")).alias("c"))
        .select(
            _chunk_id("doc_id", "c.section", "c.text").alias("chunk_id"),
            "doc_id",
            F.col("file_name").alias("source"),
            F.col("c.section").alias("section"),
            F.col("c.chunk_index").alias("chunk_index"),
            F.col("c.text").alias("text"),
            _tokens("c.text").alias("n_tokens"),
            "content_hash",
            F.current_timestamp().alias("_updated_at"),
        )
    )


def sync_chunks(spark: SparkSession, paths: Paths) -> None:
    chunks = build_chunks(spark.read.format("delta").load(paths.silver))
    if not DeltaTable.isDeltaTable(spark, paths.chunks):
        chunks.write.format("delta").save(paths.chunks)
        return
    (
        DeltaTable.forPath(spark, paths.chunks)
        .alias("t")
        .merge(chunks.alias("s"), "t.chunk_id = s.chunk_id")
        .whenNotMatchedInsertAll()
        # chunk que deixou de existir (documento editado/removido) sai da Gold
        .whenNotMatchedBySourceDelete()
        .execute()
    )


# ---------------------------------------------------------------- Embeddings
_emb_schema = T.StructType(
    [
        T.StructField("chunk_id", T.StringType()),
        T.StructField("model_name", T.StringType()),
        T.StructField("vector", T.ArrayType(T.FloatType())),
    ]
)


def _embed_partitions(batches: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    embedder = get_embedder()  # carregado uma vez por worker
    for pdf in batches:
        if pdf.empty:
            continue
        vectors = embedder.embed(pdf["text"].tolist())
        yield pd.DataFrame({"chunk_id": pdf["chunk_id"], "model_name": embedder.name, "vector": vectors})


def sync_embeddings(spark: SparkSession, paths: Paths) -> dict:
    """Embeda apenas chunks que ainda não têm vetor para o modelo atual."""
    model_name = get_embedder().name
    chunks = spark.read.format("delta").load(paths.chunks).select("chunk_id", "text")

    if DeltaTable.isDeltaTable(spark, paths.embeddings):
        existing = (
            spark.read.format("delta").load(paths.embeddings).filter(F.col("model_name") == model_name).select("chunk_id")
        )
        pending = chunks.join(existing, "chunk_id", "left_anti")
    else:
        pending = chunks

    total, todo = chunks.count(), pending.count()
    if todo:
        (
            pending.repartition(max(1, todo // 256))
            .mapInPandas(_embed_partitions, _emb_schema)
            .withColumn("dim", F.size("vector"))
            .withColumn("created_at", F.current_timestamp())
            .write.format("delta")
            .mode("append")
            .save(paths.embeddings)
        )
    return {"model": model_name, "chunks": total, "embedded_now": todo, "reused": total - todo}
