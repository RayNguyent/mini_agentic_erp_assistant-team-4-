"""Document and chunk models, plus corpus loading.

Front matter is parsed by a deliberately small scalar-only reader rather than a
YAML dependency: the corpus format is ours, the fields are flat, and the offline
profile stays dependency-free.
"""

import re
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

CORPUS_DIR = Path("docs/corpus")

# Ordered least- to most-sensitive. app/rag/acl.py maps roles onto this scale.
CLASSIFICATIONS = ("public", "internal", "finance", "restricted")

_FRONT_MATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


class Document(BaseModel):
    doc_id: str
    title: str
    project_code: str = "ALL"
    classification: str = "internal"
    owner: str = ""
    updated_at: str = ""
    source: str = ""
    path: str = ""
    text: str

    @field_validator("classification")
    @classmethod
    def known_classification(cls, value: str) -> str:
        if value not in CLASSIFICATIONS:
            raise ValueError(
                f"unknown classification '{value}' (expected one of {CLASSIFICATIONS})"
            )
        return value


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    heading: str = ""
    text: str
    classification: str = "internal"
    project_code: str = "ALL"
    updated_at: str = ""
    char_start: int = 0
    char_end: int = 0
    tokens: int = 0

    model_config = {"frozen": True}


class ScoredChunk(BaseModel):
    """A chunk plus the per-stage scores that produced its rank.

    Carrying every stage (not just the final score) is what makes the retrieval
    diagnostics inspectable: a reviewer can see whether a result was found by
    keyword, by vector, or by both.
    """

    chunk: Chunk
    score: float = 0.0
    bm25_score: float | None = None
    bm25_rank: int | None = None
    vector_score: float | None = None
    vector_rank: int | None = None
    rerank_score: float | None = None
    matched_by: list[str] = Field(default_factory=list)


def parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    """Split leading `---` front matter from the body.

    Supports flat `key: value` scalars only — no nesting, lists, or anchors.
    A document without front matter is returned as-is with empty metadata.
    """
    match = _FRONT_MATTER_RE.match(raw)
    if not match:
        return {}, raw

    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip().strip("\"'")

    return meta, raw[match.end() :]


def normalize_text(text: str) -> str:
    """Collapse the incidental formatting differences that would otherwise
    fragment retrieval: blockquote markers, hard-wrapped lines, and runs of
    blank lines. Headings and paragraph boundaries are preserved because the
    chunker depends on them."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"^>\s?", "", text, flags=re.MULTILINE)  # blockquote markers
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_document(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8")
    meta, body = parse_front_matter(raw)
    return Document(
        doc_id=meta.get("doc_id") or path.stem.upper(),
        title=meta.get("title") or path.stem,
        project_code=meta.get("project_code", "ALL"),
        classification=meta.get("classification", "internal"),
        owner=meta.get("owner", ""),
        updated_at=meta.get("updated_at", ""),
        source=meta.get("source", ""),
        path=str(path).replace("\\", "/"),
        text=normalize_text(body),
    )


def load_corpus(corpus_dir: Path | str = CORPUS_DIR) -> list[Document]:
    directory = Path(corpus_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"corpus directory not found: {directory}")
    return [load_document(path) for path in sorted(directory.glob("*.md"))]
