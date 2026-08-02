"""Token counting for chunking and context budgeting.

Uses tiktoken when it is installed and falls back to a calibrated heuristic
otherwise. The fallback matters: the deterministic offline profile must run
with no optional dependencies and no credential, and a context budget that
silently stops being enforced is worse than a slightly imprecise one.

The heuristic deliberately *over*-estimates rather than under-estimates, so a
budget miscalculation drops a low-priority context block instead of blowing
past a provider's context window mid-request.
"""

from functools import lru_cache

_CHARS_PER_TOKEN = 3.6  # conservative for English prose + markdown


@lru_cache(maxsize=8)
def _encoder(model: str):
    try:
        import tiktoken
    except ImportError:
        return None
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    if not text:
        return 0
    encoder = _encoder(model)
    if encoder is not None:
        return len(encoder.encode(text))
    return max(1, int(len(text) / _CHARS_PER_TOKEN) + 1)


def truncate_to_tokens(text: str, max_tokens: int, model: str = "gpt-4o-mini") -> str:
    """Trim `text` to fit `max_tokens`, on a word boundary where possible."""
    if max_tokens <= 0:
        return ""
    if count_tokens(text, model) <= max_tokens:
        return text

    encoder = _encoder(model)
    if encoder is not None:
        return encoder.decode(encoder.encode(text)[:max_tokens])

    budget_chars = int(max_tokens * _CHARS_PER_TOKEN)
    clipped = text[:budget_chars]
    cut = clipped.rfind(" ")
    return clipped[:cut] if cut > budget_chars // 2 else clipped


def tokenizer_name() -> str:
    """Reported by /readiness so an operator can see whether exact counting is
    active or the heuristic is standing in."""
    return "tiktoken" if _encoder("gpt-4o-mini") is not None else "heuristic"
