"""
Prompt Processing System – REST API
"""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import redis.asyncio as aioredis
import structlog
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from src.api.models import (
    Priority,
    PromptRequest,
    PromptResult,
    SubmitResponse,
    SystemStats,
    TaskStatus,
)
from src.cache.semantic_cache import SemanticCache
from src.utils.config import settings
from src.workers.celery_app import celery_app
from src.workers.tasks import process_prompt_task

log = structlog.get_logger()

# ── Lifespan ─────────────────────────────────────────────────────────────────

_redis: aioredis.Redis | None = None
_cache: SemanticCache | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _redis, _cache
    _redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    _cache = SemanticCache(redis_client=_redis)
    await _cache.initialize()
    log.info("api.startup", redis=settings.redis_url)
    yield
    await _redis.aclose()
    log.info("api.shutdown")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Prompt Processing System",
    version="1.0.0",
    description="Distributed LLM prompt processing with semantic caching and durable execution.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Dependencies ──────────────────────────────────────────────────────────────

def get_redis() -> aioredis.Redis:
    if _redis is None:
        raise RuntimeError("Redis not initialized")
    return _redis


def get_cache() -> SemanticCache:
    if _cache is None:
        raise RuntimeError("Cache not initialized")
    return _cache


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post(
    "/prompts",
    response_model=SubmitResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a prompt for processing",
)
async def submit_prompt(
    request: PromptRequest,
    redis: aioredis.Redis = Depends(get_redis),
    cache: SemanticCache = Depends(get_cache),
) -> SubmitResponse:
    """
    Accepts a prompt, checks the semantic cache, and enqueues for
    LLM processing if not cached. Returns immediately with a task_id.
    """
    # ── 1. Semantic cache check ──────────────────────────────────────────────
    if request.cache_enabled:
        hit = await cache.get(request.prompt)
        if hit:
            log.info("cache.hit", similarity=hit["similarity"])
            await _increment_stat(redis, "cache_hits")
            await _increment_stat(redis, "tasks_completed")

            # Store result directly so GET /prompts/{id} works
            task_id = f"cache-{int(time.time() * 1000)}"
            result_data = {
                "task_id": task_id,
                "status": TaskStatus.CACHED,
                "prompt": request.prompt,
                "result": hit["result"],
                "cached": True,
                "cache_similarity": hit["similarity"],
                "processing_time_ms": 0,
                "tokens_used": hit.get("tokens_used", 0),
                "model": hit.get("model"),
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
            await redis.setex(
                f"task:{task_id}",
                settings.cache_ttl_seconds,
                json.dumps(result_data),
            )

            return SubmitResponse(
                task_id=task_id,
                status=TaskStatus.CACHED,
                cached=True,
                cache_similarity=hit["similarity"],
            )

    # ── 2. Map priority → Celery queue ──────────────────────────────────────
    queue_map = {
        Priority.HIGH: "high",
        Priority.NORMAL: "default",
        Priority.LOW: "low",
    }
    queue = queue_map[request.priority]

    # ── 3. Enqueue task ──────────────────────────────────────────────────────
    task = process_prompt_task.apply_async(
        kwargs={
            "prompt": request.prompt,
            "cache_enabled": request.cache_enabled,
            "metadata": request.metadata,
        },
        queue=queue,
    )

    # ── 4. Record initial state in Redis ────────────────────────────────────
    initial = {
        "task_id": task.id,
        "status": TaskStatus.QUEUED,
        "prompt": request.prompt,
        "cached": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    await redis.setex(f"task:{task.id}", 86400, json.dumps(initial))
    await _increment_stat(redis, "tasks_submitted")

    # ── 5. Queue depth for response ──────────────────────────────────────────
    depth = await _get_queue_depth(redis)

    log.info("task.queued", task_id=task.id, queue=queue, priority=request.priority)
    return SubmitResponse(
        task_id=task.id,
        status=TaskStatus.QUEUED,
        cached=False,
        queue_position=depth,
    )


@app.get(
    "/prompts/{task_id}",
    response_model=PromptResult,
    summary="Get the result of a submitted prompt",
)
async def get_prompt_result(
    task_id: str,
    redis: aioredis.Redis = Depends(get_redis),
) -> PromptResult:
    """Poll for task result. Status will be pending → processing → completed/failed."""
    raw = await redis.get(f"task:{task_id}")

    if raw:
        data = json.loads(raw)
        return PromptResult(**data)

    # Fall back to Celery's own result backend
    async_result = celery_app.AsyncResult(task_id)
    celery_status = async_result.status

    if celery_status == "PENDING":
        return PromptResult(task_id=task_id, status=TaskStatus.QUEUED)

    if celery_status == "STARTED":
        return PromptResult(task_id=task_id, status=TaskStatus.PROCESSING)

    if celery_status == "SUCCESS":
        result = async_result.result or {}
        return PromptResult(
            task_id=task_id,
            status=TaskStatus.COMPLETED,
            result=result.get("result"),
            processing_time_ms=result.get("processing_time_ms"),
            tokens_used=result.get("tokens_used"),
            model=result.get("model"),
        )

    if celery_status == "FAILURE":
        return PromptResult(
            task_id=task_id,
            status=TaskStatus.FAILED,
            error=str(async_result.result),
        )

    raise HTTPException(status_code=404, detail=f"Task {task_id} not found")


@app.delete(
    "/prompts/{task_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Cancel a queued task",
)
async def cancel_task(
    task_id: str,
    redis: aioredis.Redis = Depends(get_redis),
) -> None:
    celery_app.control.revoke(task_id, terminate=False)

    raw = await redis.get(f"task:{task_id}")
    if raw:
        data = json.loads(raw)
        data["status"] = TaskStatus.CANCELLED
        await redis.setex(f"task:{task_id}", 3600, json.dumps(data))

    log.info("task.cancelled", task_id=task_id)


@app.get(
    "/stats",
    response_model=SystemStats,
    summary="System-wide processing statistics",
)
async def get_stats(
    redis: aioredis.Redis = Depends(get_redis),
    cache: SemanticCache = Depends(get_cache),
) -> SystemStats:
    submitted = int(await redis.get("stat:tasks_submitted") or 0)
    completed = int(await redis.get("stat:tasks_completed") or 0)
    failed = int(await redis.get("stat:tasks_failed") or 0)
    cache_hits = int(await redis.get("stat:cache_hits") or 0)

    hit_rate = (cache_hits / submitted) if submitted > 0 else 0.0

    # Active workers from Celery
    inspect = celery_app.control.inspect(timeout=1.0)
    active_map: dict[str, Any] = inspect.active() or {}
    active_count = sum(len(v) for v in active_map.values())
    worker_count = len(active_map)

    queue_depth = await _get_queue_depth(redis)
    cache_size = await cache.size()

    # Rate limit remaining (sliding window)
    import time as _time
    minute_key = f"rate:groq:{int(_time.time() // 60)}"
    used = int(await redis.get(minute_key) or 0)
    remaining = max(0, settings.groq_rate_limit - used)

    # Average processing time
    avg_raw = await redis.get("stat:total_processing_ms")
    avg_count_raw = await redis.get("stat:processing_count")
    avg_ms: float | None = None
    if avg_raw and avg_count_raw:
        count = int(avg_count_raw)
        if count > 0:
            avg_ms = round(int(avg_raw) / count, 1)

    return SystemStats(
        queue_depth=queue_depth,
        active_workers=worker_count,
        tasks_completed=completed,
        tasks_failed=failed,
        tasks_pending=max(0, submitted - completed - failed),
        cache_hit_rate=round(hit_rate, 4),
        cache_size=cache_size,
        rate_limit_remaining=remaining,
        avg_processing_time_ms=avg_ms,
    )


@app.get("/health", include_in_schema=False)
async def health(redis: aioredis.Redis = Depends(get_redis)) -> dict:
    await redis.ping()
    return {"status": "ok"}


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _increment_stat(redis: aioredis.Redis, key: str) -> None:
    await redis.incr(f"stat:{key}")


async def _get_queue_depth(redis: aioredis.Redis) -> int:
    """Approximate depth across all priority queues."""
    depths = await redis.llen("celery") or 0
    return int(depths)
