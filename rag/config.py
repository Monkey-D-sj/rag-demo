from enum import Enum
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

class SplitStrategy(Enum):
    fixed_size = "fixed_size"
    recursive_character = "recursive_character"
    paragraph_semantic = "paragraph_semantic"

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── PostgreSQL ──
    PG_HOST: str = "localhost"
    PG_PORT: int = 5432
    PG_DATABASE: str = "rag_memory"
    PG_USER: str = "rag"
    PG_PASSWORD: str = "rag123"
    PG_POOL_MIN: int = 2
    PG_POOL_MAX: int = 10

    # ── Redis ──
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str | None = None
    REDIS_MAX_CONNECTIONS: int = 10

    # ── LLM ──
    MODEL_KEY: str = ""
    MODEL_NAME: str = ""
    MODEL_URL: str = ""

    # ── Graph / 实体抽取 ──
    ENABLE_ENTITY_EXTRACTION: bool = False
    # 单篇文档内并发抽取的 chunk 数上限;调高提速但更易触发 LLM 限流。
    GRAPH_EXTRACT_CONCURRENCY: int = 4
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "neo4j_pass"
    NEO4J_DATABASE: str = "neo4j"

    # ── Embedding ──
    EMBEDDING_KEY: str = ""
    EMBEDDING_URL: str = ""
    EMBEDDING_MODEL: str = "text-embedding-v4"
    EMBEDDING_DIM: int = 1024

    # ── MinIO ──
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET: str = "rag-documents"
    MINIO_SECURE: bool = False

    # ── arq / 文档入库 ──
    ARQ_REDIS_DB: int = 1
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    SPLIT_STRATEGY: SplitStrategy = SplitStrategy.paragraph_semantic
    EMBEDDING_BATCH_SIZE: int = 16
    MAX_UPLOAD_MB: int = 20
    # 死信自愈：arq 单轮重试用尽后，cron 每 5 min 扫描 failed 文档按指数退避重试，
    # retry_count 达上限后放弃（真·死信），需人工介入。
    MAX_RETRY_ROUNDS: int = 10
    RETRY_BACKOFF_BASE: int = 60  # 秒；第 n 轮退避 = base * 2^n

    # ── Logging ──
    LOG_LEVEL: str = "INFO"                        # DEBUG / INFO / WARNING / ERROR
    LOG_FORMAT: str = "text"                       # "text"（开发） | "json"（生产）
    LOG_FILE: str | None = None                    # None=仅控制台；给路径则额外写文件
    LOG_FILE_MAX_BYTES: int = 10 * 1024 * 1024     # 单文件 10MB
    LOG_FILE_BACKUP_COUNT: int = 5                 # 轮转保留份数

    # ── Loki ──
    LOKI_ENABLED: bool = False
    LOKI_URL: str = "http://localhost:3100"
    LOKI_APP_LABEL: str = "rag-demo"

    _REQUIRED_FIELDS = (
        "MODEL_KEY", "MODEL_NAME", "MODEL_URL",
        "EMBEDDING_KEY", "EMBEDDING_URL",
    )

    def check_required(self) -> None:
        """校验必填配置项已设置；未设置则抛 ValueError，启动即失败。"""
        missing = [f for f in self._REQUIRED_FIELDS if not getattr(self, f)]
        if self.ENABLE_ENTITY_EXTRACTION:
            missing += [
                f for f in ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD", "NEO4J_DATABASE")
                if not getattr(self, f)
            ]
        if missing:
            raise ValueError(
                f"缺少必要配置: {', '.join(missing)}，请检查 .env 文件"
            )

    @property
    def PG_ASYNC_DSN(self) -> str:
        return (
            f"postgresql://{self.PG_USER}:{self.PG_PASSWORD}"
            f"@{self.PG_HOST}:{self.PG_PORT}/{self.PG_DATABASE}"
        )

    @property
    def PG_SYNC_URL(self) -> str:
        return (
            f"postgresql+psycopg://{self.PG_USER}:{self.PG_PASSWORD}"
            f"@{self.PG_HOST}:{self.PG_PORT}/{self.PG_DATABASE}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
