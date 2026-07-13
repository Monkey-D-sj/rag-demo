# Spec: 移除 DEFAULT_KB_ID，新增小说/法规双知识库

**Date**: 2026-07-13 | **Branch**: `feat/async-foundation`

## 目标

- 彻底删除 `DEFAULT_KB_ID`，不再有"默认知识库"概念
- 新增两个具名知识库：**小说**（`NOVEL_KB_ID`）和 **法规**（`REGULATION_KB_ID`）
- Chat 检索默认搜全部 KB；上传文档时必须显式选择 KB

## 变更总览

| 层 | 变更 |
|----|------|
| DB | 新 migration 0008：清旧数据，重建分区表，种子两条 KB + 分区 |
| 常量 | `DEFAULT_KB_ID` 删除，新增 `NOVEL_KB_ID` + `REGULATION_KB_ID` |
| 检索 | `kb_ids: list[str] \| None`，None=全KB，非None走 `ANY()` |
| Chat | 不传 kb_ids，默认搜全部 KB |
| 上传 API | `knowledge_base_id` 必填，无默认值 |
| 前端 | 上传区域加 KB 下拉框，硬编码小说/法规两个选项 |
| 测试 | 指向 `NOVEL_KB_ID` |

## 1. 数据层

### Migration 0008

清空旧数据并重建（分区表 + 索引结构与 0007 一致），种子数据替换为两条 KB：

```sql
INSERT INTO knowledge_bases (id, name) VALUES
  ('00000000-0000-0000-0000-000000000002', '小说'),
  ('00000000-0000-0000-0000-000000000003', '法规');
```

分区：

```sql
CREATE TABLE dchunks_novel PARTITION OF document_chunks
    FOR VALUES IN ('00000000-0000-0000-0000-000000000002');
CREATE TABLE dchunks_regulation PARTITION OF document_chunks
    FOR VALUES IN ('00000000-0000-0000-0000-000000000003');
CREATE TABLE dchunks_other PARTITION OF document_chunks DEFAULT;
```

### 常量

`rag/document/__init__.py`：

```python
NOVEL_KB_ID      = "00000000-0000-0000-0000-000000000002"
REGULATION_KB_ID = "00000000-0000-0000-0000-000000000003"
```

`ensure_kb_partition()` 保留不动 — 未来新增 KB 时 runtime 自动建分区。

## 2. 检索层

### `store.py`

`search_chunks` 和 `search_chunks_bm25` 的 `knowledge_base_id` 改为 `knowledge_base_ids: list[str] | None`：

- `None` → 省略 WHERE 子句，PG 扫全部 LIST 分区
- `list[str]` → `WHERE dc.knowledge_base_id = ANY(%(kb_ids)s)`，利用分区裁剪

### `retriever.py`

`KnowledgeRetriever.search()` 签名改为 `knowledge_base_ids: list[str] | None = None`，透传 store。

### `recall.py`

不再取 `knowledge_base_id`，调用 `retriever.search(query)` — 不传 kb_ids，搜全部。

### `workflow.py`

初始 state 删除 `"knowledge_base_id"` 字段。

## 3. 文档上传

### 后端 Controller

`knowledge_base_id` 去掉默认值，改为必填：

```python
knowledge_base_id: str = Form(...),
```

### 前端

- `client.ts`：`uploadDocument` 的 `knowledgeBaseId` 参数去掉默认值，变为必传
- UI：上传区域加下拉框 `[小说 / 法规]`，映射到对应的 KB ID 常量（前端硬编码）

## 4. 测试

| 文件 | 变更 |
|------|------|
| `tests/test_document_store.py` | `DEFAULT_KB_ID` → `NOVEL_KB_ID` |

## 5. 不变的部分

- `rag/eval/EVAL_KB_ID` 保持独立，与生产 KB 隔离
- 历史 migration（0002, 0007）不动
- `ensure_kb_partition()` 保留，供未来 KB 扩展
- `KnowledgeRetriever.search()` 仍支持 `kb_ids` 参数，eval 可限定只查评测 KB
