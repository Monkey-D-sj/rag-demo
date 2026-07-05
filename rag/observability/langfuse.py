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
    # 延迟导入,关闭时不加载 SDK
    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler

    Langfuse(
        public_key=settings.LANGFUSE_PUBLIC_KEY,
        secret_key=settings.LANGFUSE_SECRET_KEY,
        host=settings.LANGFUSE_HOST,
    )
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
