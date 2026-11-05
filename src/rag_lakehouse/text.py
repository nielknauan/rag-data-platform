"""Funções puras de texto: limpeza, mascaramento de PII, hash e chunking.

Ficam fora do Spark para serem testáveis isoladamente e reutilizadas em UDFs.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata

# --- Mascaramento de PII (governança: nada identificável entra no índice vetorial)
PII_PATTERNS: dict[str, re.Pattern] = {
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"),
    "CPF": re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b"),
    "CNPJ": re.compile(r"\b\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}\b"),
    "TELEFONE": re.compile(r"(?:\+55\s?)?\(?\b\d{2}\)?\s?9?\d{4}-?\d{4}\b"),
}


def mask_pii(text: str) -> tuple[str, dict[str, int]]:
    """Substitui PII por placeholders e devolve a contagem por tipo (para auditoria)."""
    counts: dict[str, int] = {}
    for label, pattern in PII_PATTERNS.items():
        text, n = pattern.subn(f"[{label}]", text)
        if n:
            counts[label] = n
    return text, counts


def has_pii(text: str) -> bool:
    return any(p.search(text) for p in PII_PATTERNS.values())


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def sha256(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def estimate_tokens(text: str) -> int:
    # Aproximação barata (~4 caracteres/token) para controle de tamanho e custo
    return max(1, len(text) // 4)


# --- Chunking
ARTICLE_RE = re.compile(r"(?m)^(Art\. \d+[ºo]?\.?)\s")
HEADING_RE = re.compile(r"(?m)^#{1,3} .+$")


def _split_by_size(text: str, max_chars: int, overlap: int) -> list[str]:
    """Fallback recursivo: quebra por parágrafo/linha e, se preciso, por tamanho fixo com overlap."""
    if len(text) <= max_chars:
        return [text]
    for sep in ("\n\n", "\n", ". "):
        parts = text.split(sep)
        if len(parts) > 1:
            chunks, buf = [], ""
            for part in parts:
                candidate = f"{buf}{sep}{part}" if buf else part
                if len(candidate) <= max_chars:
                    buf = candidate
                else:
                    if buf:
                        chunks.append(buf)
                    buf = part
            if buf:
                chunks.append(buf)
            out: list[str] = []
            for c in chunks:
                out.extend(_split_by_size(c, max_chars, overlap) if len(c) > max_chars else [c])
            return out
    step = max_chars - overlap
    return [text[i : i + max_chars] for i in range(0, len(text), step)]


def chunk_document(text: str, max_chars: int = 1500, overlap: int = 150) -> list[dict]:
    """Chunking orientado à estrutura.

    - Textos de lei: um chunk por artigo (a unidade que se cita).
    - Documentos com títulos markdown: um chunk por seção.
    - Seções grandes: quebra recursiva com overlap.
    Cada chunk carrega `section` (ex.: 'Art. 18') para citação e filtro.
    """
    sections: list[tuple[str, str]] = []
    if ARTICLE_RE.search(text):
        starts = [m.start() for m in ARTICLE_RE.finditer(text)]
        preamble = text[: starts[0]].strip()
        if preamble:
            sections.append(("Preâmbulo", preamble))
        for i, s in enumerate(starts):
            body = text[s : starts[i + 1] if i + 1 < len(starts) else len(text)].strip()
            num = re.search(r"\d+", ARTICLE_RE.match(body).group(1)).group(0)
            sections.append((f"Art. {num}", body))
    elif HEADING_RE.search(text):
        starts = [m.start() for m in HEADING_RE.finditer(text)]
        if starts[0] > 0:
            sections.append(("Introdução", text[: starts[0]].strip()))
        for i, s in enumerate(starts):
            body = text[s : starts[i + 1] if i + 1 < len(starts) else len(text)].strip()
            sections.append((body.splitlines()[0].lstrip("# ").strip(), body))
    else:
        sections.append(("Documento", text))

    chunks = []
    for section, body in sections:
        if not body:
            continue
        for part in _split_by_size(body, max_chars, overlap):
            part = part.strip()
            if part:
                chunks.append({"section": section, "text": part})
    for i, c in enumerate(chunks):
        c["chunk_index"] = i
    return chunks
