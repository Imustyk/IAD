"""In-memory sliding-window rate limiter (per client key)."""
from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from threading import Lock

from iad.core.exceptions import RateLimitError


@dataclass
class RateLimitConfig:
    max_requests: int = 100
    window_seconds: int = 60


class RateLimiter:
    """Thread-safe sliding window limiter.

    Production deployments should replace this with Redis (e.g. slowapi + Redis)
    when running multiple API replicas.
    """

    def __init__(self, config: RateLimitConfig | None = None) -> None:
        self.config = config or RateLimitConfig()
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def check(self, key: str, *, raise_on_limit: bool = True) -> bool:
        """Record a hit for *key*. Return False when over limit."""
        now = time.monotonic()
        window = self.config.window_seconds
        cutoff = now - window

        with self._lock:
            bucket = self._hits[key]
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            if len(bucket) >= self.config.max_requests:
                if raise_on_limit:
                    raise RateLimitError("Rate limit exceeded.")
                return False
            bucket.append(now)
            return True

    def reset(self, key: str) -> None:
        with self._lock:
            self._hits.pop(key, None)


# Process-wide singleton for middleware
_default_limiter: RateLimiter | None = None


def get_rate_limiter() -> RateLimiter:
    global _default_limiter
    if _default_limiter is None:
        from iad.config.settings import get_settings

        s = get_settings()
        _default_limiter = RateLimiter(
            RateLimitConfig(
                max_requests=s.RATE_LIMIT_REQUESTS,
                window_seconds=s.RATE_LIMIT_WINDOW_SECONDS,
            )
        )
    return _default_limiter
