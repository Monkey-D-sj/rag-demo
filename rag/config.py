from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── PostgreSQL ──
    pg_host: str = "localhost"
    pg_port: int = 5432
    pg_database: str = "rag_memory"
    pg_user: str = "rag"
    pg_password: str = "rag123"
    pg_pool_min: int = 2
    pg_pool_max: int = 10

    # ── Redis ──
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str | None = None
    redis_max_connections: int = 10

    # ── LLM ──
    model_key: str = ""
    model_name: str = ""
    model_url: str = ""

    # ── Embedding ──
    embedding_key: str = ""
    embedding_url: str = ""
    embedding_model: str = "text-embedding-v4"
    embedding_dim: int = 1024

    # ── Logging ──
    log_level: str = "INFO"                        # DEBUG / INFO / WARNING / ERROR
    log_format: str = "text"                       # "text"（开发） | "json"（生产）
    log_file: str | None = None                    # None=仅控制台；给路径则额外写文件
    log_file_max_bytes: int = 10 * 1024 * 1024     # 单文件 10MB
    log_file_backup_count: int = 5                 # 轮转保留份数

    @property
    def pg_async_dsn(self) -> str:
        return (
            f"postgresql://{self.pg_user}:{self.pg_password}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_database}"
        )

    @property
    def pg_sync_url(self) -> str:
        return (
            f"postgresql+psycopg://{self.pg_user}:{self.pg_password}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_database}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
