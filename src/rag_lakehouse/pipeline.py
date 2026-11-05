"""CLI do pipeline.

    python -m rag_lakehouse.pipeline run              # batch: landing -> índice vetorial
    python -m rag_lakehouse.pipeline stream           # streaming contínuo (Ctrl+C para parar)
    python -m rag_lakehouse.pipeline ask "pergunta"
    python -m rag_lakehouse.pipeline eval --min-hit-rate 0.7
    python -m rag_lakehouse.pipeline maintain         # OPTIMIZE + VACUUM (custo de storage)
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

from .config import get_paths

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("rag")

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "data" / "sample" / "docs"
GOLDEN = Path(__file__).resolve().parents[2] / "data" / "eval" / "golden.jsonl"


def seed_landing(paths) -> None:
    Path(paths.landing).mkdir(parents=True, exist_ok=True)
    for f in SAMPLE_DIR.glob("*"):
        target = Path(paths.landing) / f.name
        if not target.exists():
            shutil.copy(f, target)


def publish(spark, paths) -> dict:
    """Gold -> quality gates -> índice vetorial."""
    from pyspark.sql import functions as F

    from .embeddings import get_embedder
    from .gold import sync_chunks, sync_embeddings
    from .quality import run_checks
    from .vector_index import sync_index

    sync_chunks(spark, paths)
    emb_stats = sync_embeddings(spark, paths)
    dq = run_checks(spark, paths)

    model = get_embedder().name
    pdf = (
        spark.read.format("delta").load(paths.chunks)
        .join(spark.read.format("delta").load(paths.embeddings).filter(F.col("model_name") == model), "chunk_id")
        .select("chunk_id", "source", "section", "text", "vector")
        .toPandas()
    )
    idx = sync_index(pdf, paths.vector_db, model)
    return {"embeddings": emb_stats, "quality": dq, "index": idx}


def cmd_run(stream: bool, max_new_embeddings: int | None = None) -> None:
    from .ingestion import start_bronze, start_silver
    from .spark import get_spark

    paths = get_paths()
    seed_landing(paths)
    spark = get_spark()
    t0 = time.time()

    if not stream:
        start_bronze(spark, paths).awaitTermination()
        start_silver(spark, paths).awaitTermination()
        result = publish(spark, paths)
        log.info("resultado: %s", json.dumps(result, default=str))
        log.info("pipeline batch concluído em %.1fs", time.time() - t0)
        new = result["embeddings"]["embedded_now"]
        if max_new_embeddings is not None and new > max_new_embeddings:
            sys.exit(f"{new} chunks embedados; esperado no máximo {max_new_embeddings} (cache de embeddings falhou)")
        return

    log.info("streaming: solte arquivos .md/.txt em %s", paths.landing)
    bronze, silver = start_bronze(spark, paths, stream=True), start_silver(spark, paths, stream=True)
    last = None
    try:
        while True:
            time.sleep(15)
            version = spark.sql(f"DESCRIBE HISTORY delta.`{paths.silver}` LIMIT 1").first()["version"]
            if version != last:  # silver mudou -> atualiza gold e índice
                log.info("silver v%s: %s", version, json.dumps(publish(spark, paths), default=str))
                last = version
    except KeyboardInterrupt:
        bronze.stop()
        silver.stop()


def cmd_maintain() -> None:
    from delta.tables import DeltaTable

    from .spark import get_spark

    paths, spark = get_paths(), get_spark()
    spark.conf.set("spark.databricks.delta.retentionDurationCheck.enabled", "false")
    for p in (paths.bronze, paths.silver, paths.chunks, paths.embeddings):
        t = DeltaTable.forPath(spark, p)
        t.optimize().executeCompaction()  # compacta arquivos pequenos
        t.vacuum(168)  # remove arquivos órfãos com mais de 7 dias
        log.info("otimizado: %s", p)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--max-new-embeddings", type=int, default=None, help="falha se embedar mais que N chunks")
    sub.add_parser("stream")
    sub.add_parser("maintain")
    a = sub.add_parser("ask")
    a.add_argument("question")
    e = sub.add_parser("eval")
    e.add_argument("--k", type=int, default=5)
    e.add_argument("--min-hit-rate", type=float, default=0.0)
    e.add_argument("--before-incremental", action="store_true", help="ignora perguntas sobre docs da carga incremental")
    args = ap.parse_args()

    if args.cmd in ("run", "stream"):
        cmd_run(stream=args.cmd == "stream", max_new_embeddings=getattr(args, "max_new_embeddings", None))
    elif args.cmd == "maintain":
        cmd_maintain()
    elif args.cmd == "ask":
        from .rag import answer

        print(json.dumps(answer(args.question), ensure_ascii=False, indent=2))
    elif args.cmd == "eval":
        from .evaluate import run_eval

        res = run_eval(str(GOLDEN), k=args.k, include_incremental=not args.before_incremental)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        if res[f"hit_rate@{args.k}"] < args.min_hit_rate:
            sys.exit(f"hit_rate@{args.k} abaixo do mínimo ({args.min_hit_rate})")


if __name__ == "__main__":
    main()
