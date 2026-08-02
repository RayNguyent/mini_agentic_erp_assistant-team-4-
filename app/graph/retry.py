"""Bounded retry with exponential backoff and full jitter.

Two rules this module exists to enforce:

1. Only *transient* failures are retried. A bad project code or a malformed
   payload will fail identically on attempt three, so retrying it just burns
   latency and, for write-like actions, multiplies the blast radius.
2. Backoff is jittered. Synchronised retries from concurrent requests are how
   a briefly degraded provider gets turned into a fully saturated one.
"""

import random
import time
from dataclasses import dataclass
from typing import Callable, TypeVar

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Bounded retry configuration.

    `base_delay_s` is intentionally small: these are local/mock providers in the
    default profile, and the suite should not pay seconds of real sleep to prove
    the retry path works. Production profiles raise it via config.
    """

    max_retries: int = 2
    base_delay_s: float = 0.05
    max_delay_s: float = 2.0
    jitter: bool = True

    def delay_for(self, attempt: int) -> float:
        """Full-jitter backoff: ``random(0, min(cap, base * 2**attempt))``."""
        ceiling = min(self.max_delay_s, self.base_delay_s * (2**attempt))
        return random.uniform(0, ceiling) if self.jitter else ceiling


def retry_call(
    fn: Callable[[], T],
    *,
    policy: RetryPolicy,
    is_retryable: Callable[[Exception], bool],
    on_retry: Callable[[int, Exception], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[T, int]:
    """Call `fn`, retrying only exceptions `is_retryable` accepts.

    Returns ``(result, retries_used)`` so the caller can record the retry count
    in typed state instead of losing it in a log line. Re-raises the final
    exception once the budget is exhausted.
    """
    attempt = 0
    while True:
        try:
            return fn(), attempt
        except Exception as exc:  # noqa: BLE001 - re-raised unless retryable
            if not is_retryable(exc) or attempt >= policy.max_retries:
                raise
            if on_retry is not None:
                on_retry(attempt + 1, exc)
            sleep(policy.delay_for(attempt))
            attempt += 1
