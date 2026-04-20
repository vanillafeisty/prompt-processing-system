"""
Celery application configuration.

Key durability settings:
  - acks_late=True          task only ACKed after successful completion
  - reject_on_worker_lost   immediate requeue if worker process dies
  - visibility_timeout      requeue if worker silent > N seconds
  - 3 priority queues       high / default / low
"""

from celery import Celery

from src.utils.config import settings

celery_app = Celery(
    "prompt_processor",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["src.workers.tasks"],
)

celery_app.conf.update(
    # ── Serialisation ────────────────────────────────────────────────────────
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    # ── Durability ───────────────────────────────────────────────────────────
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    broker_transport_options={
        "visibility_timeout": settings.task_visibility_timeout,
        # Redis Streams: ensure messages are re-delivered if not ACKed
        "fanout_prefix": True,
        "fanout_patterns": True,
    },
    # ── Result backend TTL ───────────────────────────────────────────────────
    result_expires=86400,  # 24 hours
    # ── Concurrency ──────────────────────────────────────────────────────────
    worker_concurrency=settings.celery_concurrency,
    worker_prefetch_multiplier=1,   # fair dispatch, no hoarding
    # ── Priority queues ──────────────────────────────────────────────────────
    task_queues={
        "high":    {"exchange": "high",    "routing_key": "high"},
        "default": {"exchange": "default", "routing_key": "default"},
        "low":     {"exchange": "low",     "routing_key": "low"},
    },
    task_default_queue="default",
    task_default_exchange="default",
    task_default_routing_key="default",
    # ── Time limits ──────────────────────────────────────────────────────────
    task_soft_time_limit=120,   # SIGTERM after 2 min
    task_time_limit=180,        # SIGKILL after 3 min
    # ── Retries ──────────────────────────────────────────────────────────────
    task_max_retries=settings.max_retries,
    # ── Monitoring ───────────────────────────────────────────────────────────
    worker_send_task_events=True,
    task_send_sent_event=True,
)
