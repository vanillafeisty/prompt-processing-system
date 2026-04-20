"""
Groq LLM client with integrated rate limiter.

Each worker process maintains one GroqClient instance (lazy singleton).
The rate limiter uses a shared Redis counter to enforce the 300 req/min
limit across ALL worker processes globally.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import redis
import structlog
from groq import Groq
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.utils.config import settings
from src.utils.rate_limiter import RateLimiter

log = structlog.get_logger()

_client: "GroqClient | None" = None
_redis: redis.Redis | None = None


@dataclass
class LLMResponse:
    content: str
    model: str
    tokens_used: int
    latency_ms: int


def get_groq_client() -> "GroqClient":
    global _client, _redis
    if _client is None:
        _redis = redis.from_url(settings.redis_url)
        _client = GroqClient(redis_client=_redis)
    return _client


class GroqClient:
    def __init__(self, redis_client: redis.Redis) -> None:
        self._groq = Groq(api_key=settings.groq_api_key)
        self._rate_limiter = RateLimiter(redis_client)
        log.info("groq_client.initialized", model=settings.groq_model)

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def complete(self, prompt: str) -> LLMResponse:
        """
        Send a completion request to Groq, blocking on rate limit if needed.
        Retries up to 3 times with exponential backoff on transient errors.
        """
        # ── Acquire rate limit slot (blocks if at 300/min) ──────────────────
        self._rate_limiter.acquire(timeout=60.0)

        start = time.monotonic()
        log.info("groq.request", model=settings.groq_model, prompt_len=len(prompt))

        response = self._groq.chat.completions.create(
            model=settings.groq_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=settings.groq_max_tokens,
            temperature=0.7,
        )

        latency_ms = int((time.monotonic() - start) * 1000)
        content = response.choices[0].message.content or ""
        tokens = response.usage.total_tokens if response.usage else 0

        log.info(
            "groq.response",
            latency_ms=latency_ms,
            tokens=tokens,
            model=settings.groq_model,
        )

        return LLMResponse(
            content=content,
            model=settings.groq_model,
            tokens_used=tokens,
            latency_ms=latency_ms,
        )
