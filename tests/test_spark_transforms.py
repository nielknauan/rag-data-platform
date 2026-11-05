from datetime import datetime


def test_silver_mascara_pii_e_mantem_ultima_versao(spark):
    from rag_lakehouse.ingestion import to_silver

    rows = [
        ("versao antiga", "a.md", datetime(2026, 1, 1), datetime(2026, 1, 1)),
        ("contato: x@y.com", "a.md", datetime(2026, 2, 1), datetime(2026, 2, 1)),
        ("outro doc", "b.md", datetime(2026, 1, 1), datetime(2026, 1, 1)),
    ]
    df = spark.createDataFrame(rows, "raw_content string, file_name string, source_modified_at timestamp, _ingested_at timestamp")
    out = {r.file_name: r for r in to_silver(df).collect()}
    assert len(out) == 2
    assert out["a.md"].content == "contato: [EMAIL]" and out["a.md"].pii_masked == {"EMAIL": 1}


def test_gold_explode_chunks_com_id_estavel(spark, docs):
    from rag_lakehouse.gold import build_chunks

    df = spark.createDataFrame([("d1", "lgpd.md", docs["lgpd_lei_13709.md"], "h")],
                               "doc_id string, file_name string, content string, content_hash string")
    a = build_chunks(df).collect()
    b = build_chunks(df).collect()
    assert len(a) > 10
    assert [r.chunk_id for r in a] == [r.chunk_id for r in b]  # determinístico -> embedding reaproveitado
    assert len({r.chunk_id for r in a}) == len(a)


def test_embeddings_via_mapinpandas(spark):
    from rag_lakehouse.gold import _emb_schema, _embed_partitions

    df = spark.createDataFrame([("c1", "dados pessoais sensíveis"), ("c2", "incidente de segurança")], "chunk_id string, text string")
    rows = df.mapInPandas(_embed_partitions, _emb_schema).collect()
    assert {r.chunk_id for r in rows} == {"c1", "c2"}
    assert all(len(r.vector) == 512 and r.model_name == "hash-bow-512" for r in rows)
    pdf = df.toPandas()
    assert list(pdf.columns) == ["chunk_id", "text"]
