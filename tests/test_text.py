from rag_lakehouse.text import chunk_document, clean_text, has_pii, mask_pii


def test_mascara_pii_e_conta_por_tipo():
    txt = "Fale com joana@acme.com.br, tel (11) 91234-5678, CPF 123.456.789-09."
    masked, counts = mask_pii(txt)
    assert "joana" not in masked and "91234" not in masked and "123.456" not in masked
    assert counts == {"EMAIL": 1, "CPF": 1, "TELEFONE": 1}
    assert not has_pii(masked)


def test_nao_mascara_numero_de_lei():
    masked, counts = mask_pii("Lei nº 13.709, de 14 de agosto de 2018, art. 48")
    assert counts == {} and "13.709" in masked


def test_chunking_de_lei_e_por_artigo(docs):
    chunks = chunk_document(clean_text(docs["lgpd_lei_13709.md"]))
    sections = [c["section"] for c in chunks]
    assert {"Art. 1", "Art. 5", "Art. 7", "Art. 18", "Art. 48"} <= set(sections)
    art18 = [c for c in chunks if c["section"] == "Art. 18"]
    assert art18[0]["text"].startswith("Art. 18.")


def test_chunking_markdown_por_secao(docs):
    chunks = chunk_document(clean_text(docs["politica_privacidade_acme.md"]))
    assert "Retenção de dados" in {c["section"] for c in chunks}


def test_secao_grande_e_quebrada_com_limite():
    texto = "## Grande\n" + "\n".join(f"Linha {i} " + "x" * 80 for i in range(100))
    chunks = chunk_document(texto, max_chars=500, overlap=50)
    assert len(chunks) > 1 and all(len(c["text"]) <= 500 for c in chunks)
