from functools import lru_cache
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# 按项目根目录解析 .env，无论从哪个目录启动都能正确加载
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(_ENV_PATH), extra="ignore")

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
    NEO4J_ENABLED: bool = False        # 是否连接 Neo4j 图数据库（默认关闭）
    ENABLE_ENTITY_EXTRACTION: bool = False
    # 单篇文档内并发抽取的 chunk 数上限;调高提速但更易触发 LLM 限流。
    GRAPH_EXTRACT_CONCURRENCY: int = 4
    GRAPH_RECALL_ENABLED: bool = False  # 图召回第三路,需配合 NEO4J_ENABLED=true
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "neo4j_pass"
    NEO4J_DATABASE: str = "neo4j"

    # ── Retriever ──
    RETRIEVER_CANDIDATE_MULTIPLIER: int = 2
    RETRIEVER_VEC_SIMILARITY_THRESHOLD: float = 0.5
    RETRIEVER_RRF_K: int = 60  # RRF 融合常数，越大两路权重越均匀

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
    EMBEDDING_BATCH_SIZE: int = 10
    MAX_UPLOAD_MB: int = 20
    # 死信自愈：arq 单轮重试用尽后，cron 每 5 min 扫描 failed 文档按指数退避重试，
    # retry_count 达上限后放弃（真·死信），需人工介入。
    MAX_RETRY_ROUNDS: int = 10
    RETRY_BACKOFF_BASE: int = 60  # 秒；第 n 轮退避 = base * 2^n
    # 卡死回收阈值：pending(enqueue 丢失)或 processing(worker 超时被取消未置 failed)
    # 超过此秒数即视为卡死，由 cron 找回重投。必须 > job_timeout(300)，否则误回收运行中任务。
    STALE_DOC_SECONDS: int = 900

    # ── Logging ──
    LOG_LEVEL: str = "INFO"                        # DEBUG / INFO / WARNING / ERROR
    LOG_FORMAT: str = "text"                       # "text"（开发） | "json"（生产）
    LOG_FILE: str | None = None                    # None=仅控制台；给路径则额外写文件
    LOG_FILE_MAX_BYTES: int = 10 * 1024 * 1024     # 单文件 10MB
    LOG_FILE_BACKUP_COUNT: int = 5                 # 轮转保留份数

    # ── Rerank ──
    RERANK_ENABLED: bool = False
    RERANK_KEY: str = ""
    RERANK_BASE_URL: str = ""
    RERANK_MODEL: str = "qwen3-rerank"

    # ── Sentence Window（recall 之后、rerank 之前）──
    SENTENCE_WINDOW_ENABLED: bool = True
    SENTENCE_WINDOW_SIZE: int = 2          # 中心 chunk 左右各取 N 个邻居
    SENTENCE_WINDOW_MAX_MULTIPLIER: int = 3  # 结果最多膨胀到原始数量的 N 倍

    # ── Parent-Child Retrieval（rerank + dynamic_topk 之后）──
    PARENT_CHILD_ENABLED: bool = True

    # ── 动态 Top-K 截断（rerank 之后）──
    RERANK_DYNAMIC_TOPK_ENABLED: bool = True
    RERANK_DYNAMIC_TOPK_DEFAULT: int = 5
    RERANK_DYNAMIC_TOPK_RATIO: float = 0.7

    # ── 语义缓存(handle_query 之后,答案级,全局作用域)──
    SEMANTIC_CACHE_ENABLED: bool = False
    SEMANTIC_CACHE_SIM_THRESHOLD: float = 0.95  # 余弦相似度命中阈值
    SEMANTIC_CACHE_TTL_HOURS: int = 168         # 缓存有效期(7 天)

    # ── Loki ──
    LOKI_ENABLED: bool = False
    LOKI_URL: str = "http://localhost:3100"
    LOKI_APP_LABEL: str = "rag-demo"

    _REQUIRED_FIELDS = (
        "MODEL_KEY", "MODEL_NAME", "MODEL_URL",
        "EMBEDDING_KEY", "EMBEDDING_URL",
    )

    @model_validator(mode="after")
    def _validate_required(self):
        """校验必填配置项已设置；未设置则抛 ValueError，启动即失败。"""
        missing = [f for f in self._REQUIRED_FIELDS if not getattr(self, f)]
        if self.NEO4J_ENABLED:
            missing += [
                f for f in ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD", "NEO4J_DATABASE")
                if not getattr(self, f)
            ]
        if missing:
            raise ValueError(
                f"缺少必要配置: {', '.join(missing)}，请检查 .env 文件"
            )
        return self

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
