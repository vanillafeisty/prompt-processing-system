"""
Celery task definitions.

process_prompt_task:
  - Checks semantic cache (sync Redis lookup)
  - Calls Groq with rate limiting
  - Stores result back in cache and Redis result store
  - Retries on transient errors with exponential backoff
  - Updates task state throughout for real-time polling
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import redis
import structlog
from celery import Task
from celery.exceptions import MaxRetriesExceededError, SoftTimeLimitExceeded

from src.cache.embedder import cosine_similarity, embed
from src.utils.config import settings
from src.workers.celery_app import celery_app
from src.workers.groq_client import get_groq_client

log = structlog.get_logger()

_sync_redis: redis.Redis | None = None


def _get_redis() -> redis.Redis:
    global _sync_redis
    if _sync_redis is None:
        _sync_redis = redis.from_url(settings.redis_url, decode_responses=True)
    return _sync_redis


def _sync_cache_get(prompt: str) -> dict | None:
    """Synchronous semantic cache lookup for use inside Celery tasks."""
    r = _get_redis()
    members = r.zrange("cache:index", 0, -1)
    if not members:
        return None

    query_vec = embed(prompt)
    best_score = 0.0
    best_data: dict | None = None

    pipe = r.pipeline()
    for m in members:
        pipe.get(f"cache:meta:{m}")
    raw_entries = pipe.execute()

    for raw in raw_entries:
        if not raw:
            continue
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError:
            continue
        embedding = entry.get("embedding")
        if not embedding:
            continue
        score = cosine_similarity(query_vec, embedding)
        if score > best_score:
            best_score = score
            best_data = entry

    if best_score >= settings.cache_similarity_threshold and best_data:
        return {"result": best_data["result"], "similarity": best_score}
    return None


def _sync_cache_set(
    prompt: str, result: str, tokens_used: int = 0, model: str | None = None
) -> None:
    """Synchronous cache write for use inside Celery tasks."""
    import hashlib

    embedding = embed(prompt)
    key_hash = hashlib.sha256(prompt.encode()).hexdigest()
    entry = {
        "prompt": prompt,
        "result": result,
        "embedding": embedding,
        "tokens_used": tokens_used,
        "model": model,
        "created_at": time.time(),
    }
    r = _get_redis()
    pipe = r.pipeline()
    pipe.setex(
        f"cache:meta:{key_hash}",
        settings.cache_ttl_seconds,
        json.dumps(entry),
    )
    pipe.zadd("cache:index", {key_hash: time.time()})
    pipe.execute()


def _update_task_state(task_id: str, data: dict) -> None:
    """Persist task result to Redis for API polling."""
    r = _get_redis()
    r.setex(f"task:{task_id}", 86400, json.dumps(data))


def _record_processing_time(ms: int) -> None:
    r = _get_redis()
    pipe = r.pipeline()
    pipe.incrby("stat:total_processing_ms", ms)
    pipe.incr("stat:processing_count")
    pipe.execute()


# ── Task ──────────────────────────────────────────────────────────────────────

@celery_app.task(
    bind=True,
    name="process_prompt",
    acks_late=True,
    reject_on_worker_lost=True,
    max_retries=settings.max_retries,
    default_retry_delay=5,
)
def process_prompt_task(
    self: Task,
    prompt: str,
    cache_enabled: bool = True,
    metadata: dict | None = None,
) -> dict:
    """
    Core prompt processing task.

    Execution flow:
      1. Update state → PROCESSING
      2. Check semantic cache (sync)
      3. Acquire rate limit slot → call Groq
      4. Store result in cache
      5. Update state → COMPLETED / FAILED
    """
    task_id: str = self.request.id
    start = time.monotonic()
    log.info("task.started", task_id=task_id, prompt_len=len(prompt))

    _update_task_state(
        task_id,
        {
            "task_id": task_id,
            "status": "processing",
            "prompt": prompt,
            "cached": False,
            "started_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    try:
        # ── Cache check (worker-side) ────────────────────────────────────────
        if cache_enabled:
            hit = _sync_cache_get(prompt)
            if hit:
                elapsed = int((time.monotonic() - start) * 1000)
                log.info("task.cache_hit", task_id=task_id, similarity=hit["similarity"])
                result_data = {
                    "task_id": task_id,
                    "status": "completed",
                    "prompt": prompt,
                    "result": hit["result"],
                    "cached": True,
                    "cache_similarity": round(hit["similarity"], 4),
                    "processing_time_ms": elapsed,
                    "tokens_used": 0,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
                _update_task_state(task_id, result_data)
                _get_redis().incr("stat:tasks_completed")
                _get_redis().incr("stat:cache_hits")
                return result_data

        # ── LLM call ─────────────────────────────────────────────────────────
        client = get_groq_client()
        response = client.complete(prompt)

        elapsed = int((time.monotonic() - start) * 1000)

        # ── Cache the new result ─────────────────────────────────────────────
        if cache_enabled:
            _sync_cache_set(
                prompt,
                response.content,
                tokens_used=response.tokens_used,
                model=response.model,
            )

        # ── Record stats ─────────────────────────────────────────────────────
        _record_processing_time(elapsed)
        _get_redis().incr("stat:tasks_completed")

        result_data = {
            "task_id": task_id,
            "status": "completed",
            "prompt": prompt,
            "result": response.content,
            "cached": False,
            "processing_time_ms": elapsed,
            "tokens_used": response.tokens_used,
            "model": response.model,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        _update_task_state(task_id, result_data)

        log.info(
            "task.completed",
            task_id=task_id,
            elapsed_ms=elapsed,
            tokens=response.tokens_used,
        )
        return result_data

    except SoftTimeLimitExceeded:
        log.error("task.timeout", task_id=task_id)
        _get_redis().incr("stat:tasks_failed")
        _update_task_state(
            task_id,
            {
                "task_id": task_id,
                "status": "failed",
                "prompt": prompt,
                "error": "Task timed out after 120s",
                "failed_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        raise

    except MaxRetriesExceededError:
        log.error("task.max_retries_exceeded", task_id=task_id)
        _get_redis().incr("stat:tasks_failed")
        _update_task_state(
            task_id,
            {
                "task_id": task_id,
                "status": "failed",
                "prompt": prompt,
                "error": f"Failed after {settings.max_retries} retries",
                "failed_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        raise

    except Exception as exc:
        elapsed = int((time.monotonic() - start) * 1000)
        log.warning(
            "task.retry",
            task_id=task_id,
            error=str(exc),
            retry_count=self.request.retries,
        )

        # Exponential backoff: 5s, 25s, 125s
        countdown = 5 * (5 ** self.request.retries)

        try:
            raise self.retry(exc=exc, countdown=countdown)
        except MaxRetriesExceededError:
            _get_redis().incr("stat:tasks_failed")
            _update_task_state(
                task_id,
                {
                    "task_id": task_id,
                    "status": "failed",
                    "prompt": prompt,
                    "error": str(exc),
                    "failed_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            raise
