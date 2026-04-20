"""
Test suite for the Prompt Processing System.

Run with:
    pytest tests/ -v
    pytest tests/ -v --tb=short           # shorter tracebacks
    pytest tests/test_api.py -v -k cache  # run only cache tests
"""

from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_redis():
    """In-memory Redis mock."""
    store: dict[str, str] = {}
    scores: dict[str, float] = {}

    r = MagicMock()
    r.get = AsyncMock(side_effect=lambda k: store.get(k))
    r.setex = AsyncMock(side_effect=lambda k, ttl, v: store.update({k: v}))
    r.incr = AsyncMock(side_effect=lambda k: store.update({k: str(int(store.get(k, "0")) + 1)}) or int(store.get(k, "1")))
    r.ping = AsyncMock(return_value=True)
    r.llen = AsyncMock(return_value=0)
    r.zrange = AsyncMock(return_value=[])
    r.zcard = AsyncMock(return_value=0)
    r.zadd = AsyncMock(return_value=1)
    r.aclose = AsyncMock()

    pipe = MagicMock()
    pipe.get = MagicMock(return_value=pipe)
    pipe.setex = MagicMock(return_value=pipe)
    pipe.zadd = MagicMock(return_value=pipe)
    pipe.execute = AsyncMock(return_value=[None] * 10)
    r.pipeline = MagicMock(return_value=pipe)

    return r, store


@pytest.fixture
def mock_cache():
    cache = MagicMock()
    cache.initialize = AsyncMock()
    cache.get = AsyncMock(return_value=None)      # cache miss by default
    cache.set = AsyncMock()
    cache.size = AsyncMock(return_value=0)
    return cache


@pytest.fixture
def mock_celery_task():
    task = MagicMock()
    task.id = "test-task-id-12345"
    return task


@pytest_asyncio.fixture
async def client(mock_redis, mock_cache, mock_celery_task):
    """Async HTTP client wired to the FastAPI app with mocked dependencies."""
    from src.api.main import app, get_cache, get_redis

    r, store = mock_redis

    app.dependency_overrides[get_redis] = lambda: r
    app.dependency_overrides[get_cache] = lambda: mock_cache

    with patch("src.api.main._redis", r), \
         patch("src.api.main._cache", mock_cache):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            yield c, r, store, mock_cache

    app.dependency_overrides.clear()


# ── Health ────────────────────────────────────────────────────────────────────

class TestHealth:
    @pytest.mark.asyncio
    async def test_health_ok(self, client):
        c, *_ = client
        resp = await c.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


# ── Submit Prompt ─────────────────────────────────────────────────────────────

class TestSubmitPrompt:
    @pytest.mark.asyncio
    async def test_submit_returns_202(self, client, mock_celery_task):
        c, r, store, cache = client
        with patch("src.api.main.process_prompt_task") as mock_task:
            mock_task.apply_async.return_value = mock_celery_task
            resp = await c.post("/prompts", json={"prompt": "Hello, world!"})

        assert resp.status_code == 202
        data = resp.json()
        assert "task_id" in data
        assert data["status"] == "queued"
        assert data["cached"] is False

    @pytest.mark.asyncio
    async def test_submit_returns_cached_on_hit(self, client):
        c, r, store, cache = client
        cache.get = AsyncMock(return_value={
            "result": "42",
            "similarity": 0.97,
            "tokens_used": 10,
            "model": "llama-3.1-8b-instant",
        })
        resp = await c.post("/prompts", json={"prompt": "What is 2+2?"})

        assert resp.status_code == 202
        data = resp.json()
        assert data["status"] == "cached"
        assert data["cached"] is True
        assert data["cache_similarity"] == 0.97

    @pytest.mark.asyncio
    async def test_submit_skips_cache_when_disabled(self, client, mock_celery_task):
        c, r, store, cache = client
        cache.get = AsyncMock(return_value={"result": "cached", "similarity": 0.99})
        with patch("src.api.main.process_prompt_task") as mock_task:
            mock_task.apply_async.return_value = mock_celery_task
            resp = await c.post("/prompts", json={
                "prompt": "What is 2+2?",
                "cache_enabled": False,
            })

        assert resp.status_code == 202
        assert resp.json()["cached"] is False
        # cache should NOT have been checked
        cache.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_submit_empty_prompt_fails(self, client):
        c, *_ = client
        resp = await c.post("/prompts", json={"prompt": ""})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_submit_high_priority(self, client, mock_celery_task):
        c, r, store, cache = client
        with patch("src.api.main.process_prompt_task") as mock_task:
            mock_task.apply_async.return_value = mock_celery_task
            resp = await c.post("/prompts", json={
                "prompt": "Urgent question",
                "priority": "high",
            })
            mock_task.apply_async.assert_called_once()
            call_kwargs = mock_task.apply_async.call_args
            assert call_kwargs.kwargs["queue"] == "high"

        assert resp.status_code == 202

    @pytest.mark.asyncio
    async def test_submit_low_priority(self, client, mock_celery_task):
        c, r, store, cache = client
        with patch("src.api.main.process_prompt_task") as mock_task:
            mock_task.apply_async.return_value = mock_celery_task
            await c.post("/prompts", json={"prompt": "Low priority task", "priority": "low"})
            call_kwargs = mock_task.apply_async.call_args
            assert call_kwargs.kwargs["queue"] == "low"


# ── Get Result ────────────────────────────────────────────────────────────────

class TestGetResult:
    @pytest.mark.asyncio
    async def test_get_completed_task(self, client):
        c, r, store, cache = client
        result_data = {
            "task_id": "abc-123",
            "status": "completed",
            "prompt": "Test",
            "result": "This is the answer.",
            "cached": False,
            "processing_time_ms": 450,
            "tokens_used": 80,
            "model": "llama-3.1-8b-instant",
        }
        store["task:abc-123"] = json.dumps(result_data)
        r.get = AsyncMock(side_effect=lambda k: store.get(k))

        resp = await c.get("/prompts/abc-123")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["result"] == "This is the answer."
        assert data["processing_time_ms"] == 450

    @pytest.mark.asyncio
    async def test_get_cached_task(self, client):
        c, r, store, cache = client
        result_data = {
            "task_id": "cache-999",
            "status": "cached",
            "result": "Cached answer",
            "cached": True,
            "cache_similarity": 0.95,
        }
        store["task:cache-999"] = json.dumps(result_data)
        r.get = AsyncMock(side_effect=lambda k: store.get(k))

        resp = await c.get("/prompts/cache-999")
        assert resp.status_code == 200
        data = resp.json()
        assert data["cached"] is True
        assert data["cache_similarity"] == 0.95

    @pytest.mark.asyncio
    async def test_get_unknown_task_returns_pending(self, client):
        c, r, store, cache = client
        with patch("src.api.main.celery_app") as mock_celery:
            mock_result = MagicMock()
            mock_result.status = "PENDING"
            mock_celery.AsyncResult.return_value = mock_result
            resp = await c.get("/prompts/nonexistent-task-id")

        assert resp.status_code == 200
        assert resp.json()["status"] == "queued"


# ── Stats ─────────────────────────────────────────────────────────────────────

class TestStats:
    @pytest.mark.asyncio
    async def test_stats_returns_correct_shape(self, client):
        c, r, store, cache = client
        store.update({
            "stat:tasks_submitted": "100",
            "stat:tasks_completed": "90",
            "stat:tasks_failed": "5",
            "stat:cache_hits": "30",
            "stat:total_processing_ms": "45000",
            "stat:processing_count": "90",
        })
        r.get = AsyncMock(side_effect=lambda k: store.get(k))

        with patch("src.api.main.celery_app") as mock_celery:
            mock_celery.control.inspect.return_value.active.return_value = {
                "worker1@host": [{"id": "t1"}],
                "worker2@host": [],
            }
            resp = await c.get("/stats")

        assert resp.status_code == 200
        data = resp.json()
        assert "queue_depth" in data
        assert "cache_hit_rate" in data
        assert "rate_limit_remaining" in data
        assert 0.0 <= data["cache_hit_rate"] <= 1.0

    @pytest.mark.asyncio
    async def test_cache_hit_rate_calculation(self, client):
        c, r, store, cache = client
        store.update({
            "stat:tasks_submitted": "10",
            "stat:cache_hits": "4",
        })
        r.get = AsyncMock(side_effect=lambda k: store.get(k))

        with patch("src.api.main.celery_app") as mock_celery:
            mock_celery.control.inspect.return_value.active.return_value = {}
            resp = await c.get("/stats")

        data = resp.json()
        assert data["cache_hit_rate"] == pytest.approx(0.4, abs=0.01)


# ── Cancel Task ───────────────────────────────────────────────────────────────

class TestCancelTask:
    @pytest.mark.asyncio
    async def test_cancel_queued_task(self, client):
        c, r, store, cache = client
        store["task:to-cancel"] = json.dumps({
            "task_id": "to-cancel",
            "status": "queued",
            "prompt": "Some prompt",
        })
        r.get = AsyncMock(side_effect=lambda k: store.get(k))

        with patch("src.api.main.celery_app") as mock_celery:
            resp = await c.delete("/prompts/to-cancel")

        assert resp.status_code == 204


# ── Semantic Cache Unit Tests ─────────────────────────────────────────────────

class TestSemanticCache:
    def test_cosine_similarity_identical(self):
        from src.cache.embedder import cosine_similarity
        v = [0.5, 0.5, 0.5, 0.5]
        assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-5)

    def test_cosine_similarity_orthogonal(self):
        from src.cache.embedder import cosine_similarity
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-5)

    def test_cosine_similarity_opposite(self):
        from src.cache.embedder import cosine_similarity
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(-1.0, abs=1e-5)

    def test_embed_returns_list(self):
        from src.cache.embedder import embed
        result = embed("Hello world")
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(x, float) for x in result)

    def test_similar_prompts_high_similarity(self):
        from src.cache.embedder import cosine_similarity, embed
        a = embed("What is the capital of France?")
        b = embed("Tell me the capital city of France.")
        sim = cosine_similarity(a, b)
        assert sim > 0.85, f"Expected high similarity, got {sim}"

    def test_dissimilar_prompts_low_similarity(self):
        from src.cache.embedder import cosine_similarity, embed
        a = embed("What is the capital of France?")
        b = embed("Write a Python function to sort a list.")
        sim = cosine_similarity(a, b)
        assert sim < 0.7, f"Expected low similarity, got {sim}"


# ── Rate Limiter Unit Tests ───────────────────────────────────────────────────

class TestRateLimiter:
    def test_acquire_within_limit(self):
        from src.utils.rate_limiter import RateLimiter

        r = MagicMock()
        script = MagicMock(return_value=[1, 59])
        r.register_script = MagicMock(return_value=script)

        limiter = RateLimiter(r, limit=300)
        result = limiter.acquire()
        assert result is True

    def test_acquire_blocks_at_limit(self):
        from src.utils.rate_limiter import RateLimiter

        r = MagicMock()
        # First call: over limit (301). Second call: under limit (1 in new window).
        call_count = {"n": 0}
        def side_effect(keys, args):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return [301, 1]   # over limit, 1s left in window
            return [1, 59]         # new window, fine

        script = MagicMock(side_effect=side_effect)
        r.register_script = MagicMock(return_value=script)

        limiter = RateLimiter(r, limit=300)
        with patch("time.sleep"):
            result = limiter.acquire()
        assert result is True
        assert call_count["n"] == 2

    def test_remaining_calculation(self):
        from src.utils.rate_limiter import RateLimiter

        r = MagicMock()
        r.register_script = MagicMock(return_value=MagicMock())
        r.get = MagicMock(return_value=b"50")

        limiter = RateLimiter(r, limit=300)
        assert limiter.remaining() == 250

    def test_remaining_zero_when_exhausted(self):
        from src.utils.rate_limiter import RateLimiter

        r = MagicMock()
        r.register_script = MagicMock(return_value=MagicMock())
        r.get = MagicMock(return_value=b"350")  # over limit

        limiter = RateLimiter(r, limit=300)
        assert limiter.remaining() == 0


# ── Models Validation ─────────────────────────────────────────────────────────

class TestModels:
    def test_prompt_request_valid(self):
        from src.api.models import Priority, PromptRequest
        req = PromptRequest(prompt="Hello", priority=Priority.HIGH)
        assert req.prompt == "Hello"
        assert req.priority == Priority.HIGH
        assert req.cache_enabled is True

    def test_prompt_request_defaults(self):
        from src.api.models import Priority, PromptRequest
        req = PromptRequest(prompt="Test")
        assert req.priority == Priority.NORMAL
        assert req.cache_enabled is True
        assert req.metadata == {}

    def test_prompt_request_empty_fails(self):
        from pydantic import ValidationError
        from src.api.models import PromptRequest
        with pytest.raises(ValidationError):
            PromptRequest(prompt="")

    def test_prompt_request_too_long_fails(self):
        from pydantic import ValidationError
        from src.api.models import PromptRequest
        with pytest.raises(ValidationError):
            PromptRequest(prompt="x" * 32_001)
