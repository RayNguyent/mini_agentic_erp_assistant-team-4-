"""Demo authentication: a bearer token maps to an identified role.

Not OAuth, not a user database — the spec asks for "basic authentication,
role checks" as the baseline, and a static token map satisfies that without
pretending to be more than a demo. What matters structurally is that every
request carries an `Actor` before it reaches the runtime, and that `Actor` is
what app.security.permissions and app.rag.acl key off — nothing downstream
trusts a role the caller merely claims in the request body.
"""

import os

from app.state import Actor, Role

# token -> (user_id, role). Overridable via AUTH_TOKENS_JSON for a real demo
# deployment; the built-in defaults exist so the offline profile works with
# zero configuration.
_DEFAULT_TOKENS: dict[str, tuple[str, Role]] = {
    "dev-token": ("alice.tran", "developer"),
    "pm-token": ("bao.nguyen", "project_manager"),
    "audit-token": ("minh.pham", "auditor"),
}


class AuthError(Exception):
    """Raised for a missing or unrecognised credential. Mapped to HTTP 401 at
    the API boundary (app/api/main.py)."""


def _load_tokens() -> dict[str, tuple[str, Role]]:
    import json

    raw = os.environ.get("AUTH_TOKENS_JSON")
    if not raw:
        return dict(_DEFAULT_TOKENS)
    try:
        parsed = json.loads(raw)
        return {token: (entry["user_id"], entry["role"]) for token, entry in parsed.items()}
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"AUTH_TOKENS_JSON is malformed: {exc}") from exc


class TokenAuthenticator:
    """Resolves a bearer token to an Actor. Stateless and cheap to construct;
    kept as a class only so tests can inject a token map without env vars."""

    def __init__(self, tokens: dict[str, tuple[str, Role]] | None = None) -> None:
        self._tokens = tokens if tokens is not None else _load_tokens()

    def authenticate(self, token: str | None) -> Actor:
        if not token:
            raise AuthError("missing bearer token")
        entry = self._tokens.get(token)
        if entry is None:
            raise AuthError("unrecognised bearer token")
        user_id, role = entry
        return Actor(user_id=user_id, role=role)

    def demo_tokens(self) -> dict[str, Role]:
        """For the browser UI's role selector — token -> role only, never the
        user_id, and this must never be called with a production token map."""
        return {token: role for token, (_, role) in self._tokens.items()}


def build_default_authenticator() -> TokenAuthenticator:
    return TokenAuthenticator()
