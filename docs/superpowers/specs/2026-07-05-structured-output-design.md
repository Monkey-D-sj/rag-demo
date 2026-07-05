# 设计：实体抽取改用 Structured Output

日期：2026-07-05
状态：已批准
分支：feat/async-foundation

## 背景与目标

当前图实体/关系抽取（`rag/document/entity_extraction.py`）靠 prompt 约束模型输出 JSON，再用
`_strip_code_fence` / `_parse_result` 手工解析：剥代码块、中英文键名兼容、坏记录跳过。这条路径
解析失败率不为零，且实体 `type` 字段无校验，模型可自造类型。

目标：改用 **function calling 式 structured output**（LangChain
`with_structured_output(method="function_calling")`），让 schema 校验发生在模型调用层，
彻底移除手工 JSON 解析。选 function_calling 而非 json_schema 严格模式，是因为实际提供方
DeepSeek（`deepseek-v4-pro`）与示例提供方 DashScope qwen 都稳定支持 tool calling，而
json_schema 严格模式在 DeepSeek 上支持不确定。

## 1. 接口层：ChatModel 增加结构化调用方法

`rag/models/base.py` ABC 新增：

```python
async def ainvoke_structured(self, messages, schema: type[BaseModel]) -> BaseModel: ...
```

`rag/models/normal.py` 的 `NormalModel` 实现：

- `self._model.with_structured_output(schema, method="function_calling", include_raw=False)`
- 复用现有 `_translate` 错误翻译与 tenacity 重试骨架（3 次、指数退避抖动）。
- **新增可重试条件**：schema 校验失败（`OutputParserException` / pydantic `ValidationError`）
  也视为可重试——重新采样通常可修复。
- `astream` 不动。

## 2. Schema 定义：dataclass 换成 Pydantic

`Entity` / `Relationship` / `ExtractionResult` 从 `@dataclass` 改为 Pydantic `BaseModel`：

- 字段名用英文（`name` / `type` / `description`；`source` / `target` / `keywords` /
  `description`），function calling 参数名需 ASCII 友好。
- 每个字段带 `Field(description="<中文说明>")`，描述随 schema 传给模型，替代 prompt 中的
  字段解释。
- `Entity.type` 不用严格 enum：validator 把不在 11 类实体类型中的值兜底为 `其他`。
  理由：严格 enum 会让整次抽取因单个字段非法而重试，代价不划算；类型约束仍由 prompt
  中的类型指南承担。
- 下游 `rag/graph/aggregate.py` 按属性名访问（`ent.name`、`rel.source`），零改动。

## 3. Prompt 瘦身

删除纯粹为手写 JSON 服务的内容：

- 第 7 节「JSON 合约」
- 第 8 节「输出格式模板安全」
- `---输出格式模板---` 整段

保留语义性指令：实体类型指南、N 元关系分解、命名一致、无向去重、输出上限与优先级、
章节上下文用法、语言与专有名词。

同步删除 `_strip_code_fence`、`_parse_result` 及中英文键名兼容逻辑。

## 4. Python 侧保留的后校验

「关系两端实体必须在本次抽取的实体列表内」schema 表达不了，保留为独立小函数
（从 `_parse_result` 中拆出），抽取完成后过滤：源/目标为空或不在实体名集合中的关系丢弃。
自环（source == target）仍由 `aggregate` 层过滤。

## 5. 测试

- `ainvoke_structured`：mock 底层 ChatOpenAI/runnable，验证校验失败触发重试、错误翻译生效、
  重试耗尽后异常正确抛出。
- 抽取侧：mock ChatModel 返回 Pydantic 对象，验证关系端点过滤、空文本短路（不发起调用）、
  `type` 非法值兜底为 `其他`。
- 真实 smoke test：用 DeepSeek key 跑一次单 chunk 抽取，确认 `deepseek-v4-pro` tool calling
  配合 LangChain 正常。这是唯一无法离线验证的点。

## 风险与退路

function_calling 依赖提供方 tool calling 质量；DeepSeek 偶发 tool 参数 JSON 截断，由重试覆盖。
若 smoke test 表现不佳，退路是在 `ainvoke_structured` 同一接口内改用
`method="json_mode"` + Pydantic 校验，接口签名不变，调用方无感。

### 实施修订（2026-07-05，smoke test 后）

真实探测结果：`deepseek-v4-pro` thinking 模式拒绝强制 tool_choice（HTTP 400），
`method="json_schema"` 返回 400 "unavailable"，裸 `json_mode` 不向模型传递 schema
（字段名靠模型自猜，校验必败）。最终实现为退路的完整版：**`json_mode` + 在消息中
注入 `schema.model_json_schema()` 的 SystemMessage**（插在开头 system 消息之后）。
注入发生在 `NormalModel.ainvoke_structured` 内部，接口签名与调用方均不变。
smoke 验证通过（西游记片段：6 实体含孙悟空、6 关系、全中文、类型在词表内）。

## 非目标

- 实体命名归一（跨 chunk 别名合并）——后续单独立项。
- 关系谓词封闭词表——保持开放 keywords。
- gleaning 召回增强、黄金集评测——后续单独立项。
