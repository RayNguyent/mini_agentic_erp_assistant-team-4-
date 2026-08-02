"""Deterministic instruction-injection signals, shared by every guardrail that
needs to ask "does this text try to give the assistant a command?"

Used by app.security.injection (user input and untrusted document/tool
content, at request time) and app.memory.policy (memory-write candidates, at
write time) — one pattern list, so a new attack phrasing is fixed once
instead of drifting between two copies.

Deterministic on purpose: whether text is an attempted instruction override
must not itself depend on trusting the LLM inside the loop these patterns
protect. See OWASP LLM01 (Prompt Injection) — pattern matching is a floor, not
a complete defense; the structural defenses (role separation, the citation
gate, the approval gate) are what actually bound the damage a missed pattern
could do.
"""

import re

INSTRUCTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        # `*` (not `?`) on the qualifier group: real attempts stack qualifiers
        # ("ignore all previous instructions"), and `?` only matches
        # zero-or-one, silently missing that exact common phrasing.
        r"ignore (all |any |previous |prior )*instructions",
        r"disregard (the )?(approval|permission|citation)",
        r"skip (the )?approval",
        r"unrestricted mode",
        r"developer mode",
        r"reveal (your |the )?(api key|credential|config|system prompt)",
        r"print (your |the )?(api key|credential|config|system prompt)",
        r"always (trust|approve|allow)",
        r"never (ask|require) (for )?approval",
        r"system note:",
        r"you are now in",
        r"act as (if|though) you",
        r"forget (your |all )?(previous |prior )?(instructions|rules)",
    ]
]


def find_match(text: str) -> str | None:
    """Return the matched pattern's source, or None. A single scan point so
    every caller reports findings in the same shape."""
    for pattern in INSTRUCTION_PATTERNS:
        if pattern.search(text):
            return pattern.pattern
    return None
