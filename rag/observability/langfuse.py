from __future__ import annotations

import contextlib
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


def _tracing_active(settings: LangfuseSettings) -> bool:
    """开关与凭据齐备才算追踪激活:装饰器与 handler 必须同一判据,避免半激活状态。"""
    return bool(
        settings.LANGFUSE_ENABLED
        and settings.LANGFUSE_PUBLIC_KEY
        and settings.LANGFUSE_SECRET_KEY
    )


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
    if not _tracing_active(settings):
        if settings.LANGFUSE_ENABLED:
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
    if not _tracing_active(settings):
        return lambda fn: fn
    from langfuse import observe

    return observe(name=name)


def observe_root(name: str) -> Callable:
    """把被装饰的函数变成 trace 根 span,关闭时原样返回(零包装开销)。

    与 ``observe_if_enabled`` 逻辑等价(都是"开启时 observe(name=...),关闭时
    直通"),但语义上专用于*持有 trace 根 span*的调用点——被装饰函数体内经由
    环境 OTel 上下文创建的其它 span(LangChain CallbackHandler 的回调链、
    ``observe_if_enabled`` 装饰的检索器方法)均会嵌套进同一条 trace,而不是
    各自另起一条独立顶层 trace。调用方(如 chat service)不应直接 import
    langfuse,统一经由本模块延迟导入。
    """
    return observe_if_enabled(name)


def span_scope(name: str, input: Any = None) -> contextlib.AbstractContextManager:
    """在当前 trace 下开一个子 span,关闭时退化为 nullcontext(零开销)。

    用于给非 LangChain 管辖的内部阶段(如检索的向量/BM25 两路)在 trace 里
    留下独立节点。开启时 yield 的 span 对象支持 ``update(output=...)``;
    关闭时 yield None,调用方须做 None 防御。

    SDK 调用整体兜底:可观测性失败只能丢 span,绝不能打断业务请求
    (曾因 v3 API 名残留导致整个检索请求 500,见 spec 修订记录)。
    """
    settings = get_langfuse_settings()
    if not _tracing_active(settings):
        return contextlib.nullcontext()
    try:
        _init_client()
        from langfuse import get_client

        # v4 API:start_as_current_span 已并入 start_as_current_observation
        return get_client().start_as_current_observation(
            as_type="span", name=name, input=input
        )
    except Exception:  # noqa: BLE001 - 追踪失败降级,不拖垮调用方
        logger.warning("span_scope 创建失败,本次不记录 span: %s", name, exc_info=True)
        return contextlib.nullcontext()


def session_scope(session_id: str) -> contextlib.AbstractContextManager:
    """把 session_id 传播给当前 trace 及其内创建的所有 span。

    关闭时返回空操作的上下文管理器(零开销);开启时返回
    ``langfuse.propagate_attributes(session_id=...)`` —— 这是 SDK 4.x 文档化的
    "trace 级属性传播"机制,基于 OTel 上下文,而非依赖 LangChain 回调元数据。
    该上下文管理器只做同步的 contextvar 读写,用普通 ``with``(不是
    ``async with``)包住 async 代码块即可,不会阻塞事件循环。
    """
    settings = get_langfuse_settings()
    if not _tracing_active(settings):
        return contextlib.nullcontext()
    from langfuse import propagate_attributes

    return propagate_attributes(session_id=session_id)
