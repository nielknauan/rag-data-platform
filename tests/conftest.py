import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ["PYTHONPATH"] = str(ROOT / "src")  # workers do Spark (UDFs) importam o pacote
os.environ.setdefault("EMBEDDER", "hash")  # testes não baixam modelo


@pytest.fixture(scope="session")
def docs():
    d = ROOT / "data" / "sample" / "docs"
    return {p.name: p.read_text(encoding="utf-8") for p in d.glob("*.md")}


@pytest.fixture(scope="session")
def spark():
    from pyspark.sql import SparkSession

    s = (
        SparkSession.builder.master("local[2]").appName("tests")
        .config("spark.sql.shuffle.partitions", "2").config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    s.sparkContext.setLogLevel("ERROR")
    yield s
    s.stop()
