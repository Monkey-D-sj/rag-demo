from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from rag.common.logging import get_logger
from rag.models.base import ChatModel
from rag.prompts.entity_extraction import (
    entity_extract_prompt,
    entity_types_guidance,
    human_extract_prompt,
)

logger = get_logger()

# 单次抽取的默认上限;高价值项目少时应输出更少,不强行凑满。
DEFAULT_MAX_TOTAL_RECORDS = 40
DEFAULT_MAX_ENTITY_RECORDS = 25
DEFAULT_LANGUAGE = "English"


# 实体类型封闭集合;validator 兜底非法值,与 prompt 类型指南保持同步。
ENTITY_TYPES = frozenset({
    "Person", "Creature", "Organization", "Location", "Event", "Concept",
    "Method", "Content", "Data", "Artifact", "NaturalObject", "其他",
})


class Entity(BaseModel):
    name: str = Field(
        default="",
        description="实体名称(必填);不区分大小写时用标题大小写,整篇命名保持一致",
    )
    type: str = Field(
        default="其他",
        description="实体类型,必须是类型指南给出的类型之一;均不适用时用「其他」",
    )
    description: str = Field(
        default="", description="仅基于输入文本的实体属性与活动的简明描述"
    )

    @field_validator("name", mode="before")
    @classmethod
    def _strip_name(cls, v: object) -> str:
        return "" if v is None else str(v).strip()

    @field_validator("type", mode="before")
    @classmethod
    def _coerce_type(cls, v: object) -> str:
        v = "" if v is None else str(v).strip()
        return v if v in ENTITY_TYPES else "其他"


class Relationship(BaseModel):
    source: str = Field(default="", description="源实体名称(必填),须与实体列表中的 name 一致")
    target: str = Field(default="", description="目标实体名称(必填),须与实体列表中的 name 一致")
    keywords: str = Field(
        default="", description="概括关系总体性质、概念或主题的高级关键词,逗号分隔"
    )
    description: str = Field(
        default="", description="源实体与目标实体关系性质的简要说明"
    )

    @field_validator("source", "target", mode="before")
    @classmethod
    def _strip_endpoint(cls, v: object) -> str:
        return "" if v is None else str(v).strip()


class ExtractionResult(BaseModel):
    entities: list[Entity] = Field(
        default_factory=list, description="从输入文本提取的实体列表"
    )
    relationships: list[Relationship] = Field(
        default_factory=list, description="已提取实体之间的关系列表"
    )


def _filter_result(result: ExtractionResult) -> ExtractionResult:
    """schema 表达不了的后校验:实体名非空;关系两端必须在本次抽取的实体列表内。"""
    entities = [e for e in result.entities if e.name]
    valid_names = {e.name for e in entities}
    relationships = [
        r
        for r in result.relationships
        if r.source and r.target and r.source in valid_names and r.target in valid_names
    ]
    return ExtractionResult(entities=entities, relationships=relationships)


async def extract_entities(
    llm: ChatModel,
    text: str,
    *,
    chapter_context: str | None = None,
    language: str = DEFAULT_LANGUAGE,
    max_total_records: int = DEFAULT_MAX_TOTAL_RECORDS,
    max_entity_records: int = DEFAULT_MAX_ENTITY_RECORDS,
) -> ExtractionResult:
    """从单个 chunk 文本中抽取实体与关系,返回结构化结果。

    - llm:项目统一的 ChatModel 抽象(自带重试/超时/错误翻译)。
    - chapter_context:可选章节层级(如 "第一回 → 灵根育孕源流出"),仅作背景消歧。
    空文本直接返回空结果,不发起 LLM 调用。
    """
    if not text or not text.strip():
        return ExtractionResult()

    chapter_block = ""
    if chapter_context and chapter_context.strip():
        chapter_block = f"**---章节上下文---**\n{chapter_context.strip()}\n\n"

    chat_template = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(entity_extract_prompt),
            HumanMessagePromptTemplate.from_template(human_extract_prompt),
        ]
    )
    messages = chat_template.format_messages(
        text=text,
        chapter_block=chapter_block,
        language=language,
        max_total_records=max_total_records,
        max_entity_records=max_entity_records,
        entity_types_guidance=entity_types_guidance,
    )

    raw = await llm.ainvoke_structured(messages, ExtractionResult)
    result = _filter_result(raw)
    logger.info(
        "实体抽取完成: 实体 %d 个, 关系 %d 条",
        len(result.entities),
        len(result.relationships),
    )
    return result
