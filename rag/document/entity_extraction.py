from __future__ import annotations

import json
from pydantic import BaseModel, Field, field_validator

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from rag.common.logging import get_logger
from rag.models.base import ChatModel

logger = get_logger()

# 单次抽取的默认上限;高价值项目少时应输出更少,不强行凑满。
DEFAULT_MAX_TOTAL_RECORDS = 40
DEFAULT_MAX_ENTITY_RECORDS = 25
DEFAULT_LANGUAGE = "English"


entity_types_guidance = """
**---实体类型---**
使用以下类型之一对每个实体进行分类。如果没有合适的类型，请使用 `其他`。

- Person：人类个体，无论是真实的还是虚构的。
- Creature：非人类生物（动物、神话生物等）。
- Organization：公司、机构、政府机构、团体。
- Location：地理场所（城市、国家、建筑、区域）。
- Event：发生的事件、事故、典礼、会议。
- Concept：抽象的思想、理论、原则、信念。
- Method：程序、技术、算法、工作流程。
- Content：创意或信息类作品（书籍、文章、电影、报告）。
- Data：定量或结构化信息（统计数据、数据集、测量结果）。
- Artifact：人类创造的物理或数字对象（工具、软件、设备）。
- NaturalObject：天然非生命物体（矿物、天体、化合物）。
"""


# 注意:本模板由 LangChain 以 f-string 语法渲染,输出格式模板中的字面花括号
# 必须写成 `{{` / `}}` 转义,否则会被当成模板变量。
entity_extract_prompt = """
**---角色---**
您是一位知识图谱专家，负责从用户提示词中 `---输入文本---` 部分提取实体和关系。

**---指令---**
1.  **实体提取：**
    *   **识别：** 仅识别当前用户提示词中 `---输入文本---` 部分内定义清晰且有意义的实体。
    *   **实体详情：** 对于每个识别出的实体，提取以下信息：
        *   `名称`： 实体的名称。如果实体名称不区分大小写，则将每个重要单词的首字母大写（标题大小写）。确保在整个提取过程中**命名一致**。
        *   `类型`： 使用下方 `---实体类型---` 部分提供的类型指南对实体进行分类。如果提供的实体类型均不适用，则归类为 `其他`。
        *   `描述`： 仅根据输入文本中的信息，提供实体属性和活动的简洁而全面的描述。

2.  **关系提取：**
    *   **识别：** 识别先前提取的实体之间直接的、明确陈述的且有意义的关联。
    *   **N元关系分解：** 如果单个陈述描述了涉及两个以上实体的关系（N元关系），则将其分解为多个二元（两实体）关系对，以便单独描述。
        *   示例模式：对于“<人物1>、<人物2> 和 <人物3> 合作完成了 <项目名称>”，提取每个参与者与项目之间的二元关系，或者在合理的情况下提取参与者之间的二元关系。
    *   **关系详情：** 对于每个二元关系，提取以下字段：
        *   `源实体`： 源实体的名称。确保与实体提取的**命名一致**。如果名称不区分大小写，则将每个重要单词的首字母大写（标题大小写）。
        *   `目标实体`： 目标实体的名称。确保与实体提取的**命名一致**。如果名称不区分大小写，则将每个重要单词的首字母大写（标题大小写）。
        *   `关键词`： 一个或多个用于概括关系总体性质、概念或主题的高级关键词，用逗号分隔。
        *   `描述`： 对源实体和目标实体之间关系性质的简要解释，为其联系提供明确的理由。

3.  **关系方向与去重：**
    *   除非另有明确说明，否则将所有关系视为**无向的**。对于无向关系，交换源实体和目标实体并不构成新关系。
    *   避免输出重复的关系。

4.  **输出限制与优先级：**
    *   在此响应中，`实体` 和 `关系` 的总记录数最多为 {max_total_records} 条。
    *   在此响应中，最多输出 {max_entity_records} 个实体对象。
    *   如果高价值项目较少，则输出更少的记录。不要试图填满限制。
    *   仅输出 `源实体` 和 `目标实体` 都包含在此响应所选 `实体` 列表中的关系对象。
    *   在关系列表中，优先输出那些对输入文本核心意义**最重要**的关系。

5.  **上下文与客观性：**
    *   如果用户提示词包含 `---章节上下文---` 部分，它会给出输入文本所属的文档章节层级（例如 `h1 → h2 → h3`）。仅将其用作**背景**，以消除引用的歧义，并将实体和关系描述置于正确的上下文中。**请勿**从章节标题文本本身提取实体或关系，除非标题也出现在输入文本中，否则不要提及这些标题。
    *   确保所有实体名称和描述均使用**第三人称**书写。
    *   明确命名主语或宾语；**避免使用代词**，例如 `本文`、`本论文`、`我们公司`、`我`、`你` 和 `他/她`。

6.  **语言与专有名词：**
    *   整个输出（实体名称、关键词和描述）必须使用 {language} 编写。
    *   如果专有名词（例如，人名、地名、组织名）没有广泛接受的恰当翻译，或者翻译会导致歧义，则应保留其原始语言形式。

7.  **JSON合约：**
    *   仅返回一个包含 `实体` 和 `关系` 数组的有效 JSON 对象，不要包裹在 Markdown 代码块中，也不要添加任何解释性文字。
    *   所有字符串值必须是正确转义的 JSON 字符串（将 `"` 转义为 `\\"`，将反斜杠转义为 `\\\\`，将换行符转义为 `\\n`）。
    *   字符串值中引用的任何 LaTeX 必须使用双重转义反斜杠（例如，`\\frac` 在 JSON 中写作 `\\\\frac`）。
    *   如果达到记录限制，则立即停止添加新对象，并仅返回包含允许项目的 JSON 对象。

8.  **输出格式模板安全：**
    *   `---输出格式模板---` 部分仅包含一个输出格式模板。它**永远不是**源文本。
    *   请勿从输出格式模板中提取、推断或复制实体或关系。
    *   尖括号标记（如 `<entity_name>`）是占位符。将它们替换为从当前 `---输入文本---` 部分提取的值，切勿字面输出这些占位符。

{entity_types_guidance}

**---输出格式模板---**
{{
  "实体": [
    {{"名称": "<entity_name>", "类型": "<entity_type>", "描述": "<entity_description>"}}
  ],
  "关系": [
    {{"源实体": "<source_entity>", "目标实体": "<target_entity>", "关键词": "<keyword1, keyword2>", "描述": "<relationship_description>"}}
  ]
}}
"""


# human message:承载真正的待抽取内容,可选章节上下文放在输入文本之前作为背景。
human_extract_prompt = """{chapter_block}**---输入文本---**
{text}
"""


# 实体类型封闭集合;validator 兜底非法值,与 prompt 类型指南保持同步。
ENTITY_TYPES = frozenset({
    "Person", "Creature", "Organization", "Location", "Event", "Concept",
    "Method", "Content", "Data", "Artifact", "NaturalObject", "其他",
})


class Entity(BaseModel):
    name: str = Field(description="实体名称;不区分大小写时用标题大小写,整篇命名保持一致")
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
        return str(v or "").strip()

    @field_validator("type", mode="before")
    @classmethod
    def _coerce_type(cls, v: object) -> str:
        v = str(v or "").strip()
        return v if v in ENTITY_TYPES else "其他"


class Relationship(BaseModel):
    source: str = Field(description="源实体名称,须与实体列表中的 name 一致")
    target: str = Field(description="目标实体名称,须与实体列表中的 name 一致")
    keywords: str = Field(
        default="", description="概括关系总体性质、概念或主题的高级关键词,逗号分隔"
    )
    description: str = Field(
        default="", description="源实体与目标实体关系性质的简要说明"
    )

    @field_validator("source", "target", mode="before")
    @classmethod
    def _strip_endpoint(cls, v: object) -> str:
        return str(v or "").strip()


class ExtractionResult(BaseModel):
    entities: list[Entity] = Field(
        default_factory=list, description="从输入文本提取的实体列表"
    )
    relationships: list[Relationship] = Field(
        default_factory=list, description="已提取实体之间的关系列表"
    )


def _strip_code_fence(raw: str) -> str:
    """去掉模型偶尔加上的 ```json ... ``` 包裹,并截取首个 { 到末个 } 之间的内容。"""
    text = raw.strip()
    if text.startswith("```"):
        # 去掉首行 ```/```json 及末行 ```
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    return text


def _parse_result(raw: str) -> ExtractionResult:
    """解析模型输出的 JSON。字段兼容中英文键名,缺字段的记录直接跳过。"""
    cleaned = _strip_code_fence(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning("实体抽取输出非合法 JSON,已跳过: %s", e)
        return ExtractionResult()

    def pick(d: dict, *keys: str) -> str:
        for k in keys:
            v = d.get(k)
            if v:
                return str(v).strip()
        return ""

    result = ExtractionResult()
    for item in data.get("实体") or data.get("entities") or []:
        if not isinstance(item, dict):
            continue
        name = pick(item, "名称", "name")
        if not name:
            continue
        result.entities.append(
            Entity(
                name=name,
                type=pick(item, "类型", "type") or "其他",
                description=pick(item, "描述", "description"),
            )
        )

    valid_names = {e.name for e in result.entities}
    for item in data.get("关系") or data.get("relationships") or []:
        if not isinstance(item, dict):
            continue
        source = pick(item, "源实体", "source", "source_entity")
        target = pick(item, "目标实体", "target", "target_entity")
        # 合约要求:两端实体都必须在已抽取实体列表内,否则丢弃。
        if not source or not target or source not in valid_names or target not in valid_names:
            continue
        result.relationships.append(
            Relationship(
                source=source,
                target=target,
                keywords=pick(item, "关键词", "keywords"),
                description=pick(item, "描述", "description"),
            )
        )
    return result


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

    raw = await llm.ainvoke(messages)
    result = _parse_result(raw)
    logger.info(
        "实体抽取完成: 实体 %d 个, 关系 %d 条",
        len(result.entities),
        len(result.relationships),
    )
    return result
