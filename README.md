# Prompt Processing System

A distributed, fault-tolerant prompt processing system built with **FastAPI**, **Redis**, **Celery**, and **Groq LLM**.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                          REST API (FastAPI)                       │
│              POST /prompts  GET /prompts/{id}  GET /stats        │
└──────────────────────────────┬──────────────────────────────────┘
                                │
                    ┌───────────▼──────────┐
                    │   Semantic Cache     │  ← Redis + Embeddings
                    │  (cosine similarity) │    (skip LLM if hit)
                    └───────────┬──────────┘
                    Cache Miss  │
                    ┌───────────▼──────────┐
                    │    Task Queue        │  ← Redis Streams
                    │    (Redis + Celery)  │    (durable, ordered)
                    └───────────┬──────────┘
                                │
          ┌─────────────────────┼──────────────────┐
          │                     │                  │
   ┌──────▼──────┐      ┌───────▼──────┐   ┌──────▼──────┐
   │  Worker 1   │      │  Worker 2    │   │  Worker N   │
   │ (Celery)    │      │  (Celery)    │   │  (Celery)   │
   └──────┬──────┘      └───────┬──────┘   └──────┬──────┘
          └─────────────────────┼──────────────────┘
                                │
                    ┌───────────▼──────────┐
                    │   Rate Limiter       │  ← Token Bucket
                    │  (300 req/min Groq)  │    sliding window
                    └───────────┬──────────┘
                                │
                    ┌───────────▼──────────┐
                    │     Groq LLM API     │
                    └──────────────────────┘
```

## Key Design Decisions

| Concern | Solution | Why |
|---|---|---|
| Durable execution | Celery + Redis Streams | Tasks survive worker crashes; Redis Streams have consumer groups with acknowledgment |
| Semantic caching | Redis + sentence-transformers cosine similarity | Avoid redundant LLM calls for semantically similar prompts (e.g. "Hello" vs "Hi there") |
| Rate limiting | Token bucket per-process + Redis global counter | Respects 300 req/min Groq limit across all workers |
| Parallel processing | Celery worker pool (configurable concurrency) | Horizontal scaling, each worker handles multiple tasks |
| Crash recovery | Celery `acks_late=True` + visibility timeout | Tasks re-queued if worker dies mid-execution |

## Tech Stack

- **FastAPI** – async REST API
- **Celery** – distributed task queue with retry logic
- **Redis** – queue backend + cache + rate limiter
- **Groq** – LLM provider (llama-3.1-8b-instant)
- **sentence-transformers** – local embeddings for semantic cache
- **Docker Compose** – single-command deployment

## Quick Start

### Prerequisites
- Docker & Docker Compose
- A [Groq API key](https://console.groq.com/)

### 1. Clone & Configure

```bash
git clone https://github.com/YOUR_USERNAME/prompt-processing-system
cd prompt-processing-system
cp .env.example .env
# Edit .env and set GROQ_API_KEY
```

### 2. Start Everything

```bash
docker-compose up --build
```

This starts:
- FastAPI server on `http://localhost:8000`
- 3 Celery workers
- Redis
- Flower (Celery monitoring) on `http://localhost:5555`

### 3. Submit Prompts

```bash
# Submit a prompt
curl -X POST http://localhost:8000/prompts \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain quantum entanglement in simple terms", "priority": "high"}'

# Response:
# {"task_id": "abc-123", "status": "queued", "cached": false}

# Check result
curl http://localhost:8000/prompts/abc-123

# View system stats
curl http://localhost:8000/stats
```

## API Reference

### `POST /prompts`
Submit a prompt for processing.

**Request body:**
```json
{
  "prompt": "Your prompt text",
  "priority": "high|normal|low",      // optional, default: normal
  "cache_enabled": true,               // optional, default: true
  "metadata": {}                       // optional key-value pairs
}
```

**Response:**
```json
{
  "task_id": "uuid-string",
  "status": "queued|cached",
  "cached": false,
  "queue_position": 3
}
```

### `GET /prompts/{task_id}`
Get result of a submitted prompt.

**Response:**
```json
{
  "task_id": "uuid-string",
  "status": "pending|processing|completed|failed",
  "result": "LLM response text",
  "cached": false,
  "processing_time_ms": 843,
  "tokens_used": 127,
  "error": null
}
```

### `GET /stats`
System-wide stats.

```json
{
  "queue_depth": 12,
  "active_workers": 3,
  "tasks_completed": 1847,
  "tasks_failed": 3,
  "cache_hit_rate": 0.34,
  "rate_limit_remaining": 247,
  "avg_processing_time_ms": 620
}
```

### `DELETE /prompts/{task_id}`
Cancel a queued task.

## Semantic Caching

The cache uses **cosine similarity** on sentence embeddings to match semantically equivalent prompts:

```
"What is 2+2?"  ──embed──►  [0.23, -0.41, ...]
"How much is 2 plus 2?" ──embed──► [0.21, -0.43, ...]
                                        │
                              cosine similarity = 0.96
                              threshold = 0.92 → CACHE HIT ✓
```

Cache entries expire after **1 hour** by default (configurable).

## Rate Limiting

Uses a **sliding window** counter in Redis to enforce 300 req/min across all workers:

```
worker 1 ──────► Redis INCR rate:groq:${minute_bucket}
worker 2 ──────► Redis INCR rate:groq:${minute_bucket}
worker 3 ──────► if count > 300: sleep + retry
```

Workers that hit the limit back off with jitter and retry automatically.

## Crash Recovery

Celery is configured with:
- `acks_late=True` – task only acknowledged after successful completion
- `visibility_timeout=3600` – task requeued if worker silent for 1 hour
- `max_retries=3` with exponential backoff
- `reject_on_worker_lost=True` – immediate requeue on crash

## Scaling

```bash
# Scale to 10 workers
docker-compose up --scale worker=10

# Adjust concurrency per worker (default: 4)
CELERY_CONCURRENCY=8 docker-compose up --scale worker=5
```

## Configuration

All config via `.env`:

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | required | Groq API key |
| `REDIS_URL` | `redis://redis:6379` | Redis connection |
| `GROQ_RATE_LIMIT` | `300` | Requests per minute |
| `CACHE_SIMILARITY_THRESHOLD` | `0.92` | Cosine similarity for cache hit |
| `CACHE_TTL_SECONDS` | `3600` | Cache entry lifetime |
| `CELERY_CONCURRENCY` | `4` | Tasks per worker process |
| `MAX_RETRIES` | `3` | Max task retry attempts |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq model to use |

## Project Structure

```
prompt-processing-system/
├── src/
│   ├── api/
│   │   ├── main.py          # FastAPI app + routes
│   │   ├── models.py        # Pydantic request/response models
│   │   └── dependencies.py  # DI: Redis, Celery connections
│   ├── workers/
│   │   ├── celery_app.py    # Celery configuration
│   │   ├── tasks.py         # Task definitions + retry logic
│   │   └── groq_client.py   # Groq API wrapper + rate limiter
│   ├── cache/
│   │   ├── semantic_cache.py # Embedding cache with cosine similarity
│   │   └── embedder.py      # sentence-transformers wrapper
│   ├── queue/
│   │   └── priority_queue.py # Priority-aware task routing
│   └── utils/
│       ├── rate_limiter.py  # Redis sliding window rate limiter
│       └── logger.py        # Structured logging
├── tests/
│   ├── test_api.py
│   ├── test_cache.py
│   └── test_rate_limiter.py
├── docker/
│   ├── Dockerfile.api
│   └── Dockerfile.worker
├── docker-compose.yml
├── .env.example
└── requirements.txt
```

## Monitoring

- **Flower** (Celery): http://localhost:5555 — task queue, worker status, task history
- **Redis Insight** (optional): connect to `localhost:6379`
- **Logs**: structured JSON logs from all services

## Running Tests

```bash
# With docker
docker-compose exec api pytest tests/ -v

# Locally
pip install -r requirements.txt
pytest tests/ -v
```
