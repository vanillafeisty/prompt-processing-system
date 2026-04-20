"""
Semantic Cache using Redis.

Storage layout
──────────────
  cache:meta:{hash}   → JSON: {prompt, result, embedding, tokens_used, model, created_at}
  cache:index         → Redis Sorted Set: score=timestamp, member=hash
                        (used for LRU eviction and size queries)

On each lookup we:
  1. Fetch all cached embeddings (from the index).
  2. Compute cosine similarity with the query embedding.
  3. Return the result if similarity ≥ threshold.

For large caches this is O(N) in embeddings. For production at scale,
swap to a vector DB (Qdrant / Pinecone). At typical cache sizes (<10k)
this in-memory scan is fast enough.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from typing import Any

import structlog
import redis.asyncio as aioredis

from src.cache.embedder import cosine_similarity, embed
from src.utils.config import settings

log = structlog.get_logger()

INDEX_KEY = "cache:index"
MAX_CACHE_SIZE = 10_000


class SemanticCache:
    def __init__(
        self,
        redis_client: aioredis.Redis,
        threshold: float = settings.cache_similarity_threshold,
        ttl: int = settings.cache_ttl_seconds,
    ) -> None:
        self._redis = redis_client
        self._threshold = threshold
        self._ttl = ttl

    async def initialize(self) -> None:
        """Warm-up: nothing required for Redis-backed cache."""
        log.info("cache.initialized", threshold=self._threshold, ttl=self._ttl)

    # ── Public API ────────────────────────────────────────────────────────────

    async def get(self, prompt: str) -> dict[str, Any] | None:
        """
        Return a cached result if a semantically similar prompt exists,
        or None if no match above threshold.
        """
        query_vec = await asyncio.get_event_loop().run_in_executor(
            None, embed, prompt
        )

        # Fetch all cache keys
        members: list[str] = await self._redis.zrange(INDEX_KEY, 0, -1)
        if not members:
            return None

        best_score = 0.0
        best_data: dict | None = None

        # Batch fetch metadata
        pipe = self._redis.pipeline()
        for m in members:
            pipe.get(f"cache:meta:{m}")
        raw_entries = await pipe.execute()

        for raw in raw_entries:
            if not raw:
                continue
            try:
                entry: dict = json.loads(raw)
            except json.JSONDecodeError:
                continue

            embedding = entry.get("embedding")
            if not embedding:
                continue

            score = cosine_similarity(query_vec, embedding)
            if score > best_score:
                best_score = score
                best_data = entry

        if best_score >= self._threshold and best_data:
            log.info(
                "cache.hit",
                similarity=round(best_score, 4),
                threshold=self._threshold,
            )
            return {
                "result": best_data["result"],
                "similarity": round(best_score, 4),
                "tokens_used": best_data.get("tokens_used", 0),
                "model": best_data.get("model"),
            }

        log.debug(
            "cache.miss",
            best_similarity=round(best_score, 4),
            threshold=self._threshold,
        )
        return None

    async def set(
        self,
        prompt: str,
        result: str,
        tokens_used: int = 0,
        model: str | None = None,
    ) -> None:
        """Store a new prompt → result mapping in the cache."""
        embedding = await asyncio.get_event_loop().run_in_executor(
            None, embed, prompt
        )

        key_hash = _hash(prompt)
        entry = {
            "prompt": prompt,
            "result": result,
            "embedding": embedding,
            "tokens_used": tokens_used,
            "model": model,
            "created_at": time.time(),
        }

        pipe = self._redis.pipeline()
        pipe.setex(f"cache:meta:{key_hash}", self._ttl, json.dumps(entry))
        pipe.zadd(INDEX_KEY, {key_hash: time.time()})
        await pipe.execute()

        # Evict oldest entries if over limit
        await self._evict_if_needed()
        log.info("cache.stored", hash=key_hash[:8])

    async def size(self) -> int:
        return await self._redis.zcard(INDEX_KEY)

    async def clear(self) -> None:
        members: list[str] = await self._redis.zrange(INDEX_KEY, 0, -1)
        pipe = self._redis.pipeline()
        for m in members:
            pipe.delete(f"cache:meta:{m}")
        pipe.delete(INDEX_KEY)
        await pipe.execute()
        log.info("cache.cleared")

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _evict_if_needed(self) -> None:
        size = await self._redis.zcard(INDEX_KEY)
        if size <= MAX_CACHE_SIZE:
            return

        # Remove the oldest (lowest score = oldest timestamp)
        evict_count = size - MAX_CACHE_SIZE + 100  # remove extra buffer
        oldest = await self._redis.zrange(INDEX_KEY, 0, evict_count - 1)
        pipe = self._redis.pipeline()
        for m in oldest:
            pipe.delete(f"cache:meta:{m}")
        pipe.zremrangebyrank(INDEX_KEY, 0, evict_count - 1)
        await pipe.execute()
        log.info("cache.evicted", count=evict_count)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()
