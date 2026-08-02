"""Heading-aware chunking.

Splitting on markdown headings first, then packing paragraphs to a token
target, keeps a chunk semantically whole: a retrieved chunk about severity
ratings does not start halfway through the previous section. The heading path
travels with the chunk so a citation can say *where* in the document a claim
came from, which is most of the value of a source card.
"""

import re

from app.rag.documents import Chunk, Document
from app.tokenizer import count_tokens

TARGET_TOKENS = 320
OVERLAP_TOKENS = 60
MIN_TOKENS = 40

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def _sections(text: str) -> list[tuple[str, str, int]]:
    """Split into (heading_path, body, char_offset) triples."""
    lines = text.split("\n")
    sections: list[tuple[str, str, int]] = []
    stack: list[str] = []
    buffer: list[str] = []
    offset = 0
    section_start = 0

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            sections.append((" › ".join(stack), body, section_start))

    for line in lines:
        heading = _HEADING_RE.match(line)
        if heading:
            flush()
            buffer = []
            level = len(heading.group(1))
            stack = stack[: level - 1]
            stack.append(heading.group(2).strip())
            section_start = offset + len(line) + 1
        else:
            buffer.append(line)
        offset += len(line) + 1

    flush()
    return sections


def _pack(paragraphs: list[str]) -> list[str]:
    """Greedily pack paragraphs to TARGET_TOKENS with a trailing overlap.

    Overlap is carried as whole paragraphs rather than a sliding character
    window so a chunk boundary never lands mid-sentence.
    """
    packed: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = count_tokens(para)

        if current and current_tokens + para_tokens > TARGET_TOKENS:
            packed.append("\n\n".join(current))
            # Carry back trailing paragraphs up to the overlap budget.
            carry: list[str] = []
            carry_tokens = 0
            for prev in reversed(current):
                prev_tokens = count_tokens(prev)
                if carry_tokens + prev_tokens > OVERLAP_TOKENS:
                    break
                carry.insert(0, prev)
                carry_tokens += prev_tokens
            current = carry
            current_tokens = carry_tokens

        current.append(para)
        current_tokens += para_tokens

    if current:
        packed.append("\n\n".join(current))
    return packed


def chunk_document(document: Document) -> list[Chunk]:
    """Chunk a document, merging consecutive short sections.

    Most sections in this corpus are two or three paragraphs, well under the
    token target. Emitting one chunk per section would scatter a document into
    many ~70-token fragments, which hurts retrieval twice: BM25 length
    normalisation over-rewards tiny chunks, and the top-k budget gets spent on
    fragments too small to answer anything. So sections accumulate until the
    target is reached, and the chunk records the heading range it spans.
    """
    chunks: list[Chunk] = []
    index = 0

    buffer: list[str] = []
    buffer_headings: list[str] = []
    buffer_tokens = 0
    buffer_offset = 0

    def flush() -> None:
        nonlocal buffer, buffer_headings, buffer_tokens, buffer_offset, index
        if not buffer:
            return
        text = "\n\n".join(buffer)
        heading = buffer_headings[0] if buffer_headings else ""
        if len(buffer_headings) > 1:
            heading = f"{buffer_headings[0]} … {buffer_headings[-1]}"
        chunks.append(
            Chunk(
                chunk_id=f"{document.doc_id}#c{index:02d}",
                doc_id=document.doc_id,
                title=document.title,
                heading=heading,
                text=text,
                classification=document.classification,
                project_code=document.project_code,
                updated_at=document.updated_at,
                char_start=buffer_offset,
                char_end=buffer_offset + len(text),
                tokens=buffer_tokens,
            )
        )
        index += 1
        buffer, buffer_headings, buffer_tokens = [], [], 0

    for heading, body, offset in _sections(document.text):
        paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
        if not paragraphs:
            continue

        for piece in _pack(paragraphs):
            piece_tokens = count_tokens(piece)
            if buffer and buffer_tokens + piece_tokens > TARGET_TOKENS:
                flush()
            if not buffer:
                buffer_offset = offset
            buffer.append(f"## {heading}\n{piece}" if heading else piece)
            buffer_tokens += piece_tokens
            if heading and heading not in buffer_headings:
                buffer_headings.append(heading)

    flush()

    # A trailing scrap gets folded back rather than left as its own chunk.
    if len(chunks) > 1 and chunks[-1].tokens < MIN_TOKENS:
        tail = chunks.pop()
        previous = chunks[-1]
        chunks[-1] = previous.model_copy(
            update={
                "text": f"{previous.text}\n\n{tail.text}",
                "tokens": previous.tokens + tail.tokens,
                "char_end": tail.char_end,
            }
        )

    return chunks


def chunk_corpus(documents: list[Document]) -> list[Chunk]:
    return [chunk for document in documents for chunk in chunk_document(document)]
