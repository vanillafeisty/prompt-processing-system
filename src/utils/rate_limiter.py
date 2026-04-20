"""
Redis-backed sliding window rate limiter.

Enforces a per-minute request cap across ALL worker processes by using
a shared Redis key per minute bucket. Workers that exceed the limit
back off with randomised jitter and retry.
"""

from __future__ import annotations

import random
import time

import redis
import structlog

from src.utils.config import settings

log = structlog.get_logger()

# Lua script for atomic check-and-increment
# Returns (current_count, ttl_remaining)
_CHECK_AND_INCREMENT = """
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local window = tonumber(ARGV[2])

local current = redis.call('INCR', key)
if current == 1 then
    redis.call('EXPIRE', key, window)
end

local ttl = redis.call('TTL', key)
return {current, ttl}
"""


class RateLimiter:
    """
    Sliding window rate limiter backed by Redis.
    Thread-safe: each call is atomic via a Lua script.
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        limit: int = settings.groq_rate_limit,
        window_seconds: int = 60,
    ) -> None:
        self._redis = redis_client
        self._limit = limit
        self._window = window_seconds
        self._script = redis_client.register_script(_CHECK_AND_INCREMENT)

    def acquire(self, *, timeout: float = 60.0) -> bool:
        """
        Block until a slot is available within `timeout` seconds.
        Returns True if acquired, raises RuntimeError on timeout.
        """
        deadline = time.monotonic() + timeout
        attempts = 0

        while time.monotonic() < deadline:
            key = f"rate:groq:{int(time.time() // self._window)}"
            count, ttl = self._script(keys=[key], args=[self._limit, self._window])

            if count <= self._limit:
                log.debug("rate_limiter.acquired", count=count, limit=self._limit)
                return True

            # Over limit — wait until next window or TTL, with jitter
            wait = max(0.1, min(ttl or 1, 2.0)) + random.uniform(0, 0.5)
            attempts += 1
            log.warning(
                "rate_limiter.throttled",
                count=count,
                limit=self._limit,
                wait_s=round(wait, 2),
                attempt=attempts,
            )
            time.sleep(wait)

        raise RuntimeError(
            f"Rate limit acquire timed out after {timeout}s "
            f"(limit={self._limit}/min)"
        )

    def remaining(self) -> int:
        """Return how many requests are available in the current window."""
        key = f"rate:groq:{int(time.time() // self._window)}"
        used = int(self._redis.get(key) or 0)
        return max(0, self._limit - used)
