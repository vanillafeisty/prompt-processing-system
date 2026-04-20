# Testing Guide

## Overview

Three levels of testing are available:

| Method | Speed | Requires Docker | Best for |
|---|---|---|---|
| Unit tests (pytest) | Fast (~10s) | No | Logic, cache, rate limiter |
| Integration tests (curl) | Medium | Yes | API contract, full flow |
| Load test (script) | Slow | Yes | Rate limiting, parallel workers |

---

## 1. Unit Tests (Pytest)

### Setup (local, no Docker needed)

```bash
# Create virtual env
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install deps
pip install -r requirements.txt

# Run all tests
pytest tests/ -v

# Run specific test class
pytest tests/test_api.py::TestSemanticCache -v

# Run only cache-related tests
pytest tests/ -v -k "cache"

# Run with coverage report
pip install pytest-cov
pytest tests/ --cov=src --cov-report=term-missing
```

### Expected output

```
tests/test_api.py::TestHealth::test_health_ok                    PASSED
tests/test_api.py::TestSubmitPrompt::test_submit_returns_202     PASSED
tests/test_api.py::TestSubmitPrompt::test_submit_returns_cached  PASSED
tests/test_api.py::TestSubmitPrompt::test_skip_cache_disabled    PASSED
tests/test_api.py::TestSubmitPrompt::test_empty_prompt_fails     PASSED
tests/test_api.py::TestSubmitPrompt::test_high_priority          PASSED
tests/test_api.py::TestGetResult::test_get_completed_task        PASSED
tests/test_api.py::TestSemanticCache::test_similar_prompts       PASSED
tests/test_api.py::TestRateLimiter::test_acquire_within_limit    PASSED
...
```

---

## 2. Integration Tests (Docker + curl)

### Start the system

```bash
cp .env.example .env
# Edit .env — set your GROQ_API_KEY

docker-compose up --build
# Wait for "Application startup complete" in api logs
```

### Test 1 — Health check

```bash
curl http://localhost:8000/health
# Expected: {"status":"ok"}
```

### Test 2 — Submit a prompt

```bash
curl -s -X POST http://localhost:8000/prompts \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain what a binary search tree is in 2 sentences"}' | jq .
```

Expected response:
```json
{
  "task_id": "3f8a1b2c-...",
  "status": "queued",
  "cached": false,
  "queue_position": 0
}
```

### Test 3 — Poll for result

```bash
# Replace TASK_ID with the id from Test 2
TASK_ID="3f8a1b2c-..."

curl -s http://localhost:8000/prompts/$TASK_ID | jq .
```

Poll every 2 seconds until status = "completed":
```bash
watch -n 2 "curl -s http://localhost:8000/prompts/$TASK_ID | jq '.status,.result'"
```

### Test 4 — Semantic cache hit

Submit the same prompt (or a similar one):

```bash
# First call — hits LLM
curl -s -X POST http://localhost:8000/prompts \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is a binary search tree?"}' | jq .

# Slight variation — should hit cache (similarity ~0.95)
curl -s -X POST http://localhost:8000/prompts \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Can you explain binary search trees?"}' | jq .
```

Second response should show `"status": "cached"` and a `cache_similarity` score.

### Test 5 — Priority queues

```bash
# Submit one of each priority
curl -s -X POST http://localhost:8000/prompts \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Low priority task", "priority": "low"}' | jq .task_id

curl -s -X POST http://localhost:8000/prompts \
  -H "Content-Type: application/json" \
  -d '{"prompt": "High priority task", "priority": "high"}' | jq .task_id
```

High priority tasks will be processed first.

### Test 6 — System stats

```bash
curl -s http://localhost:8000/stats | jq .
```

Expected:
```json
{
  "queue_depth": 0,
  "active_workers": 3,
  "tasks_completed": 5,
  "tasks_failed": 0,
  "cache_hit_rate": 0.4,
  "cache_size": 3,
  "rate_limit_remaining": 295,
  "avg_processing_time_ms": 620.5
}
```

### Test 7 — Cancel a task

```bash
# Submit a task then immediately cancel it
TASK_ID=$(curl -s -X POST http://localhost:8000/prompts \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Cancel me immediately", "priority": "low"}' | jq -r .task_id)

curl -s -X DELETE http://localhost:8000/prompts/$TASK_ID
# Expected: HTTP 204 No Content
```

### Test 8 — Dashboard UI

Open in your browser: **http://localhost:8000**

You should see:
- Live stat cards updating every 4 seconds
- The rate limit bar
- A prompt submission form
- A live activity feed

---

## 3. Load Test (Rate Limiting)

This script fires 50 prompts in parallel to verify the rate limiter holds at 300/min.

```bash
# Install httpie or use the Python script below
python tests/load_test.py
```

Create `tests/load_test.py`:

```python
import asyncio
import httpx
import time

API = "http://localhost:8000"
NUM_REQUESTS = 50
PROMPTS = [
    f"What is {i} multiplied by {i+1}?" for i in range(NUM_REQUESTS)
]

async def submit(client, prompt, idx):
    start = time.monotonic()
    r = await client.post(f"{API}/prompts", json={"prompt": prompt})
    elapsed = round((time.monotonic() - start) * 1000)
    data = r.json()
    print(f"[{idx:02d}] {data['status']:8s} cached={data['cached']}  {elapsed}ms")
    return data

async def main():
    async with httpx.AsyncClient(timeout=30) as client:
        tasks = [submit(client, p, i) for i, p in enumerate(PROMPTS)]
        results = await asyncio.gather(*tasks)

    cached = sum(1 for r in results if r.get("cached"))
    queued = sum(1 for r in results if r.get("status") == "queued")
    print(f"\nSummary: {queued} queued, {cached} cached out of {NUM_REQUESTS} requests")

asyncio.run(main())
```

```bash
python tests/load_test.py
```

---

## 4. Worker Crash Recovery Test

Verify tasks re-queue after a worker dies:

```bash
# 1. Submit a batch of tasks
for i in $(seq 1 10); do
  curl -s -X POST http://localhost:8000/prompts \
    -H "Content-Type: application/json" \
    -d "{\"prompt\": \"Task number $i — what is the meaning of life?\"}" | jq .task_id
done

# 2. Kill a worker mid-processing
docker-compose kill worker     # kills all workers

# 3. Check tasks are still in queue
curl -s http://localhost:8000/stats | jq .queue_depth

# 4. Restart workers — tasks should resume
docker-compose up worker -d

# 5. Verify tasks completed
curl -s http://localhost:8000/stats | jq .tasks_completed
```

---

## 5. Monitoring (Flower)

Open **http://localhost:5555** for the Celery dashboard.

Shows:
- Active / completed / failed tasks in real time
- Per-worker task history
- Task retry counts
- Queue depths by priority

---

## Common Issues

| Issue | Fix |
|---|---|
| `Connection refused` on port 8000 | Wait 10s for containers to start, check `docker-compose logs api` |
| Tasks stuck in `queued` | Check workers are running: `docker-compose ps` |
| `GROQ_API_KEY` error | Set key in `.env` file |
| Cache never hits | Similarity threshold too high — lower `CACHE_SIMILARITY_THRESHOLD` in `.env` |
| `ModuleNotFoundError` in unit tests | Run `pip install -r requirements.txt` in your venv |
