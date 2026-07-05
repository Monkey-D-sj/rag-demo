# 实体抽取 Structured Output 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实体/关系抽取从「prompt 约束 JSON + 手工解析」改为 function calling 式 structured output，schema 校验发生在模型调用层。

**Architecture:** `ChatModel` ABC 新增 `ainvoke_structured(messages, schema)`，`NormalModel` 用 LangChain `with_structured_output(method="function_calling")` 实现并复用现有 tenacity 重试/错误翻译；`entity_extraction.py` 的 dataclass 换成 Pydantic 模型（字段描述随 schema 传给模型），删除手工 JSON 解析，prompt 删除 JSON 合约相关段落；关系端点校验保留为 Python 侧后过滤。

**Tech Stack:** Python 3.12+ / pydantic v2 / langchain-openai>=1.3.0 / tenacity / pytest (asyncio_mode=auto) / uv

**Spec:** `docs/superpowers/specs/2026-07-05-structured-output-design.md`

## Global Constraints

- 所有命令用 `uv run` 前缀执行（uv 管理的项目）。
- pytest 配置了 `asyncio_mode = "auto"`：async 测试函数不需要 `@pytest.mark.asyncio` 装饰器。
- 除 Task 4 的 smoke 脚本外，测试一律不触网、不依赖真实 API key。
- 中文注释风格与现有代码一致（半角逗号，注释解释「为什么」）。
- 下游 `rag/graph/aggregate.py` 与 `rag/graph/pipeline.py` 的生产代码零改动（按属性名访问，兼容）。
- 每个 Task 结束时全量测试必须通过：`uv run pytest -q`。

---

### Task 1: `ChatModel.ainvoke_structured` 接口与 `NormalModel` 实现

**Files:**
- Modify: `rag/models/base.py`
- Modify: `rag/models/normal.py`
- Test: `tests/test_normal.py`

**Interfaces:**
- Consumes: 现有 `NormalModel._translate`、`rag.common.exception.is_retryable`。
- Produces: `async def ainvoke_structured(self, messages: list[Any], schema: type[BaseModel]) -> BaseModel` —— Task 3 的 `extract_entities` 调用它。校验失败（`OutputParserException` / pydantic `ValidationError`）可重试（最多 3 次）；模型未调用工具（结果为 None）抛 `OutputParserException`；`LLMNonRetryableError` 立即传播。

- [ ] **Step 1: 写失败测试**

在 `tests/test_normal.py` 现有内容之后追加（并在文件头部补充 import）：

```python
import pytest
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, ValidationError
from tenacity import wait_none

import rag.models.normal as normal
from rag.common.exception import LLMAuthenticationError
```

```python
class _Out(BaseModel):
    x: int


def _validation_error() -> ValidationError:
    try:
        _Out(x="不是数字")
    except ValidationError as e:
        return e
    raise AssertionError("unreachable")


class _FakeStructured:
    """按序返回预设结果;Exception 项则抛出,None 项模拟模型未调用工具。"""

    def __init__(self, results):
        self._results = list(results)
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        item = self._results.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class _FakeChatModel:
    def __init__(self, structured):
        self._structured = structured

    def with_structured_output(self, schema, method):
        assert method == "function_calling"
        return self._structured


def _model_with(results):
    m = NormalModel(Settings())
    fake = _FakeStructured(results)
    m._model = _FakeChatModel(fake)
    return m, fake


async def test_ainvoke_structured_returns_schema_instance():
    m, fake = _model_with([_Out(x=1)])
    result = await m.ainvoke_structured(["hi"], _Out)
    assert result == _Out(x=1)
    assert fake.calls == 1


async def test_validation_error_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr(normal, "wait_exponential_jitter", lambda **kw: wait_none())
    m, fake = _model_with([_validation_error(), _Out(x=2)])
    result = await m.ainvoke_structured(["hi"], _Out)
    assert result == _Out(x=2)
    assert fake.calls == 2


async def test_none_result_raises_after_exhausting_retries(monkeypatch):
    monkeypatch.setattr(normal, "wait_exponential_jitter", lambda **kw: wait_none())
    m, fake = _model_with([None, None, None])
    with pytest.raises(OutputParserException):
        await m.ainvoke_structured(["hi"], _Out)
    assert fake.calls == 3


async def test_non_retryable_llm_error_propagates_immediately():
    m, fake = _model_with([LLMAuthenticationError("bad key", status_code=401)])
    with pytest.raises(LLMAuthenticationError):
        await m.ainvoke_structured(["hi"], _Out)
    assert fake.calls == 1
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_normal.py -v`
Expected: 新增 4 个测试 FAIL，报 `AttributeError: 'NormalModel' object has no attribute 'ainvoke_structured'`；原有 `test_ainvoke_returns_content` 仍 PASS。

- [ ] **Step 3: 实现**

`rag/models/base.py` 全文替换为：

```python
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class ChatModel(ABC):
    @abstractmethod
    async def ainvoke(self, messages: list[Any]) -> str: ...

    @abstractmethod
    async def ainvoke_structured(
        self, messages: list[Any], schema: type[BaseModel]
    ) -> BaseModel:
        """结构化输出调用:返回 schema 的实例,由实现方保证 schema 校验。"""
        ...

    @abstractmethod
    def astream(self, messages: list[Any]):
        """async generator of chunks"""
        ...
```

`rag/models/normal.py` 头部新增 import：

```python
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, ValidationError
```

模块级（`dispatch_error` 之前）新增重试谓词：

```python
def _is_retryable_structured(exc: BaseException) -> bool:
    """结构化输出重试谓词:schema 解析/校验失败重新采样通常可修复,也视为可重试。"""
    if isinstance(exc, (OutputParserException, ValidationError)):
        return True
    return is_retryable(exc)
```

`NormalModel` 类中、`ainvoke` 之后新增方法：

```python
    async def ainvoke_structured(
        self, messages: list[BaseMessage | str], schema: type[BaseModel]
    ) -> BaseModel:
        """带重试的结构化输出调用:function_calling 方式强制模型按 schema 返回。"""
        structured = self._model.with_structured_output(
            schema, method="function_calling"
        )
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential_jitter(initial=1, max=10, jitter=1),
            retry=retry_if_exception(_is_retryable_structured),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        ):
            with attempt:
                async with self._translate():
                    result = await structured.ainvoke(messages)
                    if result is None:
                        raise OutputParserException("模型未返回结构化输出(未调用工具)")
                    return result
        raise AssertionError("unreachable")
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_normal.py -v`
Expected: 5 个测试全部 PASS。

- [ ] **Step 5: 全量测试 + 提交**

Run: `uv run pytest -q`
Expected: 全部 PASS。

```bash
git add rag/models/base.py rag/models/normal.py tests/test_normal.py
git commit -m "feat: ChatModel 新增 ainvoke_structured 结构化输出接口"
```

---

### Task 2: 抽取 schema 从 dataclass 迁移到 Pydantic

**Files:**
- Modify: `rag/document/entity_extraction.py`（仅 `Entity` / `Relationship` / `ExtractionResult` 三个类及相关 import）
- Modify: `tests/test_graph_pipeline.py`（位置参数构造改关键字）
- Modify: `tests/test_graph_aggregate.py`（同上）
- Test: `tests/test_entity_extraction.py`（新建）

**Interfaces:**
- Consumes: 无（独立改动；此时 `_parse_result` 仍在，用关键字构造，兼容）。
- Produces: Pydantic 模型 `Entity(name, type="其他", description="")`、`Relationship(source, target, keywords="", description="")`、`ExtractionResult(entities=[], relationships=[])`，字段名与默认值如上；`ENTITY_TYPES` 常量。Task 3 把 `ExtractionResult` 作为 schema 传给 `ainvoke_structured`。

- [ ] **Step 1: 写失败测试**

新建 `tests/test_entity_extraction.py`：

```python
from rag.document.entity_extraction import Entity, Relationship


def test_entity_type_outside_taxonomy_coerced_to_other():
    assert Entity(name="X", type="Superhero").type == "其他"


def test_entity_type_in_taxonomy_kept():
    assert Entity(name="X", type="Person").type == "Person"


def test_entity_name_and_endpoints_stripped():
    assert Entity(name=" 孙悟空 ").name == "孙悟空"
    r = Relationship(source=" A ", target=" B ")
    assert (r.source, r.target) == ("A", "B")
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_entity_extraction.py -v`
Expected: FAIL——dataclass 版 `Entity` 缺 `type`/`description` 位置参数报 `TypeError`，且无类型兜底。

- [ ] **Step 3: 实现**

`rag/document/entity_extraction.py`：

删除 `from dataclasses import dataclass, field`，新增：

```python
from pydantic import BaseModel, Field, field_validator
```

三个 dataclass（`Entity` / `Relationship` / `ExtractionResult`）整体替换为：

```python
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
```

`tests/test_graph_pipeline.py`：两处 `fake_extract`（约 42-46 行与 86-92 行）中的位置参数构造改为关键字。两处的返回值都改成：

```python
        return ExtractionResult(
            entities=[
                Entity(name="孙悟空", type="Person"),
                Entity(name="唐僧", type="Person"),
            ],
            relationships=[
                Relationship(source="孙悟空", target="唐僧", keywords="师徒"),
            ],
        )
```

（第二处 `if "BOOM" in text: raise RuntimeError("llm down")` 的前置逻辑保留不动。）

`tests/test_graph_aggregate.py` 全文替换为：

```python
from rag.document.entity_extraction import Entity, ExtractionResult, Relationship
from rag.graph.aggregate import aggregate


def _res(entities, rels):
    return ExtractionResult(entities=entities, relationships=rels)


def test_merges_same_name_across_chunks_unions_chunk_ids():
    c0 = _res([Entity(name="孙悟空", type="Person", description="d")], [])
    c1 = _res([Entity(name="孙悟空", type="Person", description="d2")], [])
    ents, rels = aggregate([("doc:0", c0), ("doc:1", c1)])
    assert len(ents) == 1
    assert ents[0]["name"] == "孙悟空"
    assert sorted(ents[0]["chunk_ids"]) == ["doc:0", "doc:1"]


def test_type_conflict_uses_majority_then_first():
    c0 = _res([Entity(name="X", type="Person", description="d")], [])
    c1 = _res([Entity(name="X", type="Creature", description="d")], [])
    c2 = _res([Entity(name="X", type="Person", description="d")], [])
    ents, _ = aggregate([("doc:0", c0), ("doc:1", c1), ("doc:2", c2)])
    assert ents[0]["type"] == "Person"  # 众数


def test_relation_direction_normalized_and_keywords_unioned():
    c0 = _res(
        [Entity(name="唐僧", type="Person"), Entity(name="孙悟空", type="Person")],
        [Relationship(source="孙悟空", target="唐僧", keywords="师徒")],
    )
    c1 = _res(
        [Entity(name="唐僧", type="Person"), Entity(name="孙悟空", type="Person")],
        [Relationship(source="唐僧", target="孙悟空", keywords="取经")],
    )
    _, rels = aggregate([("doc:0", c0), ("doc:1", c1)])
    assert len(rels) == 1
    r = rels[0]
    assert (r["source"], r["target"]) == ("唐僧", "孙悟空")  # 字典序规范化
    assert sorted(r["keywords"]) == ["取经", "师徒"]


def test_drops_relation_with_unknown_endpoint():
    c0 = _res(
        [Entity(name="A", type="Person")],
        [Relationship(source="A", target="B", keywords="k")],
    )
    ents, rels = aggregate([("doc:0", c0)])
    assert [e["name"] for e in ents] == ["A"]
    assert rels == []
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_entity_extraction.py tests/test_graph_aggregate.py tests/test_graph_pipeline.py -v`
Expected: 全部 PASS。

- [ ] **Step 5: 全量测试 + 提交**

Run: `uv run pytest -q`
Expected: 全部 PASS。

```bash
git add rag/document/entity_extraction.py tests/test_entity_extraction.py tests/test_graph_aggregate.py tests/test_graph_pipeline.py
git commit -m "refactor: 抽取 schema 从 dataclass 迁移到 Pydantic,type 非法值兜底为其他"
```

---

### Task 3: `extract_entities` 切换 structured output，prompt 瘦身，删除手工解析

**Files:**
- Modify: `rag/document/entity_extraction.py`
- Test: `tests/test_entity_extraction.py`（追加）

**Interfaces:**
- Consumes: Task 1 的 `llm.ainvoke_structured(messages, ExtractionResult)`；Task 2 的 Pydantic 模型。
- Produces: `extract_entities` 签名不变（`llm, text, *, chapter_context, language, max_total_records, max_entity_records`），返回 `ExtractionResult`；调用方 `rag/graph/pipeline.py` 零改动。

- [ ] **Step 1: 写失败测试**

`tests/test_entity_extraction.py` 头部 import 改为：

```python
from rag.document.entity_extraction import (
    Entity,
    ExtractionResult,
    Relationship,
    extract_entities,
)
```

文件末尾追加：

```python
class _FakeLLM:
    """只实现 ainvoke_structured 的鸭子类型 ChatModel。"""

    def __init__(self, result):
        self._result = result
        self.calls = 0

    async def ainvoke_structured(self, messages, schema):
        assert schema is ExtractionResult
        self.calls += 1
        return self._result


async def test_empty_text_short_circuits_without_llm_call():
    llm = _FakeLLM(ExtractionResult())
    result = await extract_entities(llm, "   ")
    assert result == ExtractionResult()
    assert llm.calls == 0


async def test_relation_with_unknown_endpoint_filtered():
    llm = _FakeLLM(
        ExtractionResult(
            entities=[Entity(name="孙悟空", type="Person")],
            relationships=[
                Relationship(source="孙悟空", target="唐僧", keywords="师徒"),
            ],
        )
    )
    result = await extract_entities(llm, "text")
    assert [e.name for e in result.entities] == ["孙悟空"]
    assert result.relationships == []


async def test_happy_path_passes_through():
    llm = _FakeLLM(
        ExtractionResult(
            entities=[
                Entity(name="孙悟空", type="Person"),
                Entity(name="唐僧", type="Person"),
            ],
            relationships=[
                Relationship(source="孙悟空", target="唐僧", keywords="师徒"),
            ],
        )
    )
    result = await extract_entities(llm, "text")
    assert len(result.entities) == 2
    assert len(result.relationships) == 1
    assert llm.calls == 1
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_entity_extraction.py -v`
Expected: 新增 3 个测试 FAIL——现 `extract_entities` 调用 `llm.ainvoke`（fake 没有该方法，报 `AttributeError`）；空文本用例因短路逻辑已存在会 PASS。

- [ ] **Step 3: 实现**

`rag/document/entity_extraction.py`：

1. 删除 `import json`。
2. `entity_extract_prompt` 上方的转义说明注释（`# 注意:本模板由 LangChain 以 f-string 语法渲染...` 两行）删除——输出格式模板移除后 prompt 不再含字面花括号。
3. `entity_extract_prompt` 全文替换为：

```python
entity_extract_prompt = """
**---角色---**
您是一位知识图谱专家，负责从用户提示词中 `---输入文本---` 部分提取实体和关系，并通过结构化输出返回结果。

**---指令---**
1.  **实体提取：**
    *   **识别：** 仅识别当前用户提示词中 `---输入文本---` 部分内定义清晰且有意义的实体。
    *   **实体字段：** 对于每个识别出的实体，填写以下字段：
        *   `name`： 实体的名称。如果实体名称不区分大小写，则将每个重要单词的首字母大写（标题大小写）。确保在整个提取过程中**命名一致**。
        *   `type`： 使用下方 `---实体类型---` 部分提供的类型指南对实体进行分类。如果提供的实体类型均不适用，则归类为 `其他`。
        *   `description`： 仅根据输入文本中的信息，提供实体属性和活动的简洁而全面的描述。

2.  **关系提取：**
    *   **识别：** 识别先前提取的实体之间直接的、明确陈述的且有意义的关联。
    *   **N元关系分解：** 如果单个陈述描述了涉及两个以上实体的关系（N元关系），则将其分解为多个二元（两实体）关系对，以便单独描述。
        *   示例模式：对于「<人物1>、<人物2> 和 <人物3> 合作完成了 <项目名称>」，提取每个参与者与项目之间的二元关系，或者在合理的情况下提取参与者之间的二元关系。
    *   **关系字段：** 对于每个二元关系，填写以下字段：
        *   `source`： 源实体的名称，须与实体列表中的 `name` **命名一致**。
        *   `target`： 目标实体的名称，须与实体列表中的 `name` **命名一致**。
        *   `keywords`： 一个或多个用于概括关系总体性质、概念或主题的高级关键词，用逗号分隔。
        *   `description`： 对源实体和目标实体之间关系性质的简要解释，为其联系提供明确的理由。

3.  **关系方向与去重：**
    *   除非另有明确说明，否则将所有关系视为**无向的**。对于无向关系，交换源实体和目标实体并不构成新关系。
    *   避免输出重复的关系。

4.  **输出限制与优先级：**
    *   在此响应中，实体和关系的总记录数最多为 {max_total_records} 条。
    *   在此响应中，最多输出 {max_entity_records} 个实体对象。
    *   如果高价值项目较少，则输出更少的记录。不要试图填满限制。
    *   仅输出 `source` 和 `target` 都包含在此响应所选实体列表中的关系对象。
    *   在关系列表中，优先输出那些对输入文本核心意义**最重要**的关系。

5.  **上下文与客观性：**
    *   如果用户提示词包含 `---章节上下文---` 部分，它会给出输入文本所属的文档章节层级（例如 `h1 → h2 → h3`）。仅将其用作**背景**，以消除引用的歧义，并将实体和关系描述置于正确的上下文中。**请勿**从章节标题文本本身提取实体或关系，除非标题也出现在输入文本中，否则不要提及这些标题。
    *   确保所有实体名称和描述均使用**第三人称**书写。
    *   明确命名主语或宾语；**避免使用代词**，例如 `本文`、`本论文`、`我们公司`、`我` 、`你` 和 `他/她`。

6.  **语言与专有名词：**
    *   整个输出（实体名称、关键词和描述）必须使用 {language} 编写。
    *   如果专有名词（例如，人名、地名、组织名）没有广泛接受的恰当翻译，或者翻译会导致歧义，则应保留其原始语言形式。

{entity_types_guidance}
"""
```

4. 删除 `_strip_code_fence` 与 `_parse_result` 两个函数，原位置新增：

```python
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
```

5. `extract_entities` 中，原来的两行

```python
    raw = await llm.ainvoke(messages)
    result = _parse_result(raw)
```

替换为：

```python
    raw = await llm.ainvoke_structured(messages, ExtractionResult)
    result = _filter_result(raw)
```

docstring 中「自带重试/超时/错误翻译」的描述仍准确，不动。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_entity_extraction.py -v`
Expected: 6 个测试全部 PASS。

- [ ] **Step 5: 全量测试 + 提交**

Run: `uv run pytest -q`
Expected: 全部 PASS（`test_graph_pipeline.py` patch 的是 `gp.extract_entities`，不受影响）。

```bash
git add rag/document/entity_extraction.py tests/test_entity_extraction.py
git commit -m "feat: 实体抽取切换 function_calling 结构化输出,移除手工 JSON 解析"
```

---

### Task 4: 真实 smoke test（DeepSeek）+ 收尾验证

**Files:**
- Create: `<scratchpad>/smoke_extraction.py`（临时脚本，不入库）

**Interfaces:**
- Consumes: Task 1-3 全部成果 + 本机 `.env` 的 DeepSeek 配置。
- Produces: `deepseek-v4-pro` tool calling 配合 `with_structured_output(method="function_calling")` 可用性的实证。这是设计中唯一无法离线验证的点。

- [ ] **Step 1: 写 smoke 脚本**

写入 scratchpad 目录（session 的 scratchpad 路径，不进 git）：

```python
import asyncio

from rag.config import Settings
from rag.document.entity_extraction import extract_entities
from rag.models.normal import NormalModel

TEXT = (
    "孙悟空拜菩提祖师为师,学得七十二般变化。"
    "后来孙悟空大闹天宫,被如来佛祖压在五行山下。"
)


async def main() -> None:
    llm = NormalModel(Settings())
    result = await extract_entities(llm, TEXT, language="中文")
    print(f"实体 {len(result.entities)} 个:")
    for e in result.entities:
        print(f"  [{e.type}] {e.name}: {e.description}")
    print(f"关系 {len(result.relationships)} 条:")
    for r in result.relationships:
        print(f"  {r.source} -> {r.target} ({r.keywords}): {r.description}")


asyncio.run(main())
```

- [ ] **Step 2: 运行 smoke 脚本**

Run: `uv run python <scratchpad>/smoke_extraction.py`
Expected: 无 traceback；输出若干实体（应含「孙悟空」，type 在 11 类之内）和至少 1 条关系，全中文。

若失败（DeepSeek 不支持 tool calling / 参数截断频发）：按 spec 退路，把 `NormalModel.ainvoke_structured` 中 `method="function_calling"` 改为 `method="json_mode"`，重跑 Task 1 测试（fake 中 `assert method == "function_calling"` 同步改）与本 smoke 脚本。

- [ ] **Step 3: 全量回归**

Run: `uv run pytest -q`
Expected: 全部 PASS。

- [ ] **Step 4: 汇报结果**

smoke 输出贴给用户确认抽取质量（实体命名、类型分布是否合理），不自动提交任何额外代码。
