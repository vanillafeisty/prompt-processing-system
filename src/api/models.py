from __future__ import annotations

from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class Priority(str, Enum):
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class TaskStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CACHED = "cached"
    CANCELLED = "cancelled"


# ── Requests ─────────────────────────────────────────────────────────────────

class PromptRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=32_000)
    priority: Priority = Priority.NORMAL
    cache_enabled: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Responses ────────────────────────────────────────────────────────────────

class SubmitResponse(BaseModel):
    task_id: str
    status: TaskStatus
    cached: bool = False
    cache_similarity: float | None = None
    queue_position: int | None = None


class PromptResult(BaseModel):
    task_id: str
    status: TaskStatus
    prompt: str | None = None
    result: str | None = None
    cached: bool = False
    processing_time_ms: int | None = None
    tokens_used: int | None = None
    model: str | None = None
    error: str | None = None
    created_at: str | None = None
    completed_at: str | None = None


class SystemStats(BaseModel):
    queue_depth: int
    active_workers: int
    tasks_completed: int
    tasks_failed: int
    tasks_pending: int
    cache_hit_rate: float
    cache_size: int
    rate_limit_remaining: int
    avg_processing_time_ms: float | None = None
