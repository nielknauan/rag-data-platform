"""Bronze e Silver com Structured Streaming.

O mesmo código roda em batch (`trigger(availableNow=True)`: processa o que chegou e
para) ou em streaming contínuo (`trigger(processingTime=...)`). Checkpoints garantem
exactly-once: um arquivo nunca é ingerido duas vezes.
"""
from __future__ import annotations

from delta.tables import DeltaTable
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T

from .config import Paths
from .text import clean_text, mask_pii, sha256


def _trigger(stream: bool) -> dict:
    return {"processingTime": "10 seconds"} if stream else {"availableNow": True}


# ---------------------------------------------------------------- Bronze
def start_bronze(spark: SparkSession, paths: Paths, stream: bool = False):
    """landing/*.md|txt -> bronze.documents (Delta, append-only, conteúdo bruto)."""
    raw = (
        spark.readStream.format("text")
        .option("wholetext", True)
        .option("pathGlobFilter", "*.{md,txt}")
        .option("maxFilesPerTrigger", 100)
        .load(paths.landing)
    )
    bronze = raw.select(
        F.col("value").alias("raw_content"),
        F.col("_metadata.file_path").alias("source_path"),
        F.col("_metadata.file_name").alias("file_name"),
        F.col("_metadata.file_modification_time").alias("source_modified_at"),
        F.current_timestamp().alias("_ingested_at"),
    )
    return (
        bronze.writeStream.format("delta")
        .option("checkpointLocation", paths.checkpoint("bronze"))
        .outputMode("append")
        .trigger(**_trigger(stream))
        .start(paths.bronze)
    )


# ---------------------------------------------------------------- Silver
_clean_schema = T.StructType(
    [
        T.StructField("content", T.StringType()),
        T.StructField("pii_masked", T.MapType(T.StringType(), T.IntegerType())),
    ]
)


@F.udf(_clean_schema)
def _clean_and_mask(raw: str):
    text, counts = mask_pii(clean_text(raw or ""))
    return text, counts


@F.udf(T.StringType())
def _hash(text: str) -> str:
    return sha256(text or "")


def to_silver(batch: DataFrame) -> DataFrame:
    """Limpa, mascara PII e mantém só a versão mais recente de cada documento."""
    w = Window.partitionBy("doc_id").orderBy(F.col("source_modified_at").desc(), F.col("_ingested_at").desc())
    return (
        batch.withColumn("doc_id", _hash("file_name"))
        .withColumn("_rn", F.row_number().over(w))
        .filter("_rn = 1")
        .withColumn("c", _clean_and_mask("raw_content"))
        .select(
            "doc_id",
            "file_name",
            F.col("c.content").alias("content"),
            F.col("c.pii_masked").alias("pii_masked"),
            _hash(F.col("c.content")).alias("content_hash"),
            "source_modified_at",
            F.current_timestamp().alias("_updated_at"),
        )
    )


def start_silver(spark: SparkSession, paths: Paths, stream: bool = False):
    """bronze (stream) -> silver.documents via foreachBatch + MERGE (upsert idempotente)."""

    def upsert(batch: DataFrame, _batch_id: int) -> None:
        if batch.isEmpty():
            return
        docs = to_silver(batch)
        if not DeltaTable.isDeltaTable(spark, paths.silver):
            docs.write.format("delta").save(paths.silver)
            return
        (
            DeltaTable.forPath(spark, paths.silver)
            .alias("t")
            .merge(docs.alias("s"), "t.doc_id = s.doc_id")
            # Só reescreve quando o conteúdo mudou de fato: evita re-chunking/re-embedding
            .whenMatchedUpdateAll(condition="t.content_hash <> s.content_hash")
            .whenNotMatchedInsertAll()
            .execute()
        )

    return (
        spark.readStream.format("delta")
        .load(paths.bronze)
        .writeStream.foreachBatch(upsert)
        .option("checkpointLocation", paths.checkpoint("silver"))
        .trigger(**_trigger(stream))
        .start()
    )
