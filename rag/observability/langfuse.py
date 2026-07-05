from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable

from pydantic_settings import BaseSettings, SettingsConfigDict

from rag.common.logging import get_logger

logger = get_logger()


class LangfuseSettings(BaseSettings):
    """Langfuse 独立配置。

    不并入 rag.config.Settings,可观测性关注点独立成模块,也避免与并行改动的
    config.py 冲突(见设计文档)。变量名与 langfuse SDK 原生环境变量同名。
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    LANGFUSE_ENABLED: bool = False
    LANGFUSE_HOST: str = "http://localhost:3000"
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""


@lru_cache
def get_langfuse_settings() -> LangfuseSettings:
    return LangfuseSettings()


@lru_cache
def _init_client() -> None:
    """进程内只初始化一次全局 Langfuse 客户端。

    Langfuse() 构造会建立底层 HTTP client/后台线程等昂贵资源,
    每次请求都重建没有必要;用 lru_cache 让其在进程生命周期内只跑一次,
    后续调用直接命中缓存(无参数,缓存 key 恒定)。CallbackHandler 仍需
    每次请求新建一个(它不是线程安全的单例,需绑定当次调用)。
    """
    settings = get_langfuse_settings()
    # 延迟导入,关闭时不加载 SDK
    from langfuse import Langfuse

    Langfuse(
        public_key=settings.LANGFUSE_PUBLIC_KEY,
        secret_key=settings.LANGFUSE_SECRET_KEY,
        host=settings.LANGFUSE_HOST,
    )


def get_callback_handler() -> Any | None:
    """开启时返回 LangChain CallbackHandler,关闭时返回 None(零开销)。

    凭据可能只写在 .env 而未导出为进程环境变量,SDK 读不到,
    所以这里显式初始化全局客户端而不是依赖 SDK 自读环境。
    """
    settings = get_langfuse_settings()
    if not settings.LANGFUSE_ENABLED:
        return None
    if not (settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY):
        logger.warning("LANGFUSE_ENABLED=true 但缺少 key,追踪已跳过")
        return None
    _init_client()
    # 延迟导入,关闭时不加载 SDK
    from langfuse.langchain import CallbackHandler

    return CallbackHandler()


def observe_if_enabled(name: str) -> Callable:
    """开启时用 langfuse.observe 包装函数,关闭时原样返回(零包装开销)。

    装饰器在 import 时求值,LANGFUSE_ENABLED 需在进程启动前设定。
    """
    settings = get_langfuse_settings()
    if not settings.LANGFUSE_ENABLED:
        return lambda fn: fn
    from langfuse import observe

    return observe(name=name)
