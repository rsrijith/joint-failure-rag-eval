"""Retry-on-rate-limit decorator for API judges.

Wraps the network call only — does not retry on parse errors or anything
that wouldn't change between attempts.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

# Substrings in str(exception) that mark the call as retryable.
_RETRYABLE_TOKENS = (
    "rate limit",
    "rate_limit",
    "429",
    "queue_exceeded",
    "too many requests",
    "service unavailable",
    "503",
    "504",
    "overloaded",
)


def retry_on_rate_limit(
    max_retries: int = 4,
    initial_delay: float = 2.0,
    max_delay: float = 60.0,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Retry up to max_retries times with exponential backoff on rate-limit errors."""

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapped(*args, **kwargs) -> T:
            delay = initial_delay
            last_exc: BaseException | None = None
            for attempt in range(max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    msg = str(e).lower()
                    is_retryable = any(tok in msg for tok in _RETRYABLE_TOKENS)
                    if is_retryable and attempt < max_retries:
                        time.sleep(delay)
                        delay = min(delay * 2, max_delay)
                        last_exc = e
                        continue
                    raise
            assert last_exc is not None
            raise last_exc

        return wrapped

    return decorator
