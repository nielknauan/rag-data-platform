import pandas as pd

from rag_lakehouse.embeddings import HashEmbedder
from rag_lakehouse.evaluate import hit_and_rank, metrics
from rag_lakehouse.text import chunk_document, clean_text, mask_pii, sha256
from rag_lakehouse.vector_index import search, sync_index


def _frame(docs, emb):
    rows = []
    for name, raw in docs.items():
        text, _ = mask_pii(clean_text(raw))
        for c in chunk_document(text):
            rows.append({"chunk_id": sha256(name, c["section"], c["text"]), "source": name,
                         "section": c["section"], "text": c["text"]})
    df = pd.DataFrame(rows)
    df["vector"] = emb.embed(df["text"].tolist())
    return df


def test_embedder_normalizado_e_deterministico():
    e = HashEmbedder(64)
    a, b = e.embed(["dados pessoais"])[0], e.embed(["dados pessoais"])[0]
    assert a == b and abs(sum(x * x for x in a) - 1) < 1e-6


def test_indice_sincroniza_incremental_e_busca(tmp_path, docs):
    emb = HashEmbedder()
    df = _frame(docs, emb)
    info = sync_index(df, str(tmp_path), emb.name)
    assert info["rows"] == len(df)

    hits = search(str(tmp_path), emb.name, emb.embed(["o controlador deverá comunicar à autoridade nacional e ao titular a ocorrência de incidente de segurança"])[0], k=5)
    assert "Art. 48" in hits["section"].tolist()

    # documento editado: um chunk sai -> some do índice
    info = sync_index(df.iloc[1:], str(tmp_path), emb.name)
    assert info["rows"] == len(df) - 1

    # filtro por metadado
    only_acme = search(str(tmp_path), emb.name, emb.embed(["prazo"])[0], k=3,
                       where="source = 'politica_privacidade_acme.md'")
    assert set(only_acme["source"]) == {"politica_privacidade_acme.md"}


def test_metricas_de_avaliacao():
    got = [("a", "Art. 1"), ("a", "Art. 5")]
    assert hit_and_rank(got, ("a", "Art. 5")) == 2
    m = metrics([1, 2, None, 1], k=5)
    assert m["hit_rate@5"] == 0.75 and m["mrr@5"] == round((1 + 0.5 + 1) / 4, 3)
