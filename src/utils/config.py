from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Groq
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"
    groq_rate_limit: int = 300        # requests per minute
    groq_max_tokens: int = 1024

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Cache
    cache_similarity_threshold: float = 0.92
    cache_ttl_seconds: int = 3600

    # Celery
    celery_concurrency: int = 4
    max_retries: int = 3
    task_visibility_timeout: int = 3600

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"


settings = Settings()
