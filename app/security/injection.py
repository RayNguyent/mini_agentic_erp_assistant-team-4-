"""Prompt-injection guardrails applied at request time.

Two distinct checks, because they mean different things:

- `screen_user_input`: the caller is authenticated, so a match here is a
  policy signal (log it, flag risk_level) rather than an automatic block —
  a legitimate user is allowed to ask "how do I skip an approval step" as a
  question about policy without being shut out.
- `screen_untrusted_content`: retrieved documents, tool output, and web
  results have no authenticated author. A match here is always flagged as
  untrusted and never allowed to change control flow — the caller of this
  function is expected to keep the content in the prompt only inside an
  explicit untrusted-evidence wrapper (see app.rag.answer.build_evidence_prompt)
  and to never let it set approval_required=False or skip the citation gate.

Neither function can enforce anything on its own — flagging is not blocking.
The actual enforcement is structural, elsewhere: the approval gate
(app.runtime.execute_write_tool) does not read from prompt output, and the
citation gate (app.rag.answer) does not accept a claim of sufficiency without
evidence. This module's job is visibility — an attempted injection shows up in
the audit log and the trace — not the sole line of defense.
"""

from pydantic import BaseModel

from app.security.patterns import find_match


class InjectionFinding(BaseModel):
    flagged: bool
    matched_pattern: str | None = None
    excerpt: str = ""


def _excerpt(text: str, around: int = 40) -> str:
    return text.strip()[: around * 2] + ("…" if len(text.strip()) > around * 2 else "")


def screen_user_input(text: str) -> InjectionFinding:
    matched = find_match(text)
    return InjectionFinding(flagged=matched is not None, matched_pattern=matched, excerpt=_excerpt(text) if matched else "")


def screen_untrusted_content(text: str, source: str = "document") -> InjectionFinding:
    matched = find_match(text)
    return InjectionFinding(flagged=matched is not None, matched_pattern=matched, excerpt=_excerpt(text) if matched else "")


def screen_chunks(chunks) -> list[tuple[str, InjectionFinding]]:
    """Screen a batch of retrieved chunks, returning only the flagged ones as
    (chunk_id, finding) pairs — used to populate the audit trail without
    logging every clean chunk."""
    findings = []
    for chunk in chunks:
        finding = screen_untrusted_content(chunk.text, source="document")
        if finding.flagged:
            findings.append((chunk.chunk_id, finding))
    return findings
