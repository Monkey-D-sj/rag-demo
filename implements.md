# rag-demo 实现方案

> 本文档记录 `improvements.md` 中改进项的具体实现方案。当前只展开 **P1-1 语义缓存 session 隔离**；其余条目待本项确认后逐项补充。

## P1-1 语义缓存 session 隔离

### 1. 目标

把当前答案级语义缓存从全局作用域改为会话作用域，保证：

1. 缓存查找、近重复检查、写入和命中计数均限定在同一个 `session_id`。
2. 不同 session 即使提出完全相同的问题、得到完全相同的 embedding，也绝不互相命中。
3. 同一 session 内的相似问题仍可按现有阈值命中。
4. `session_id` 缺失时关闭本次缓存能力，绝不降级成全局查询。
5. 会话删除后，其缓存自动删除；文档入库后的全局缓存失效与 TTL 清理保持不变。

### 2. 已确认的设计结论

| 决策 | 结论 | 原因 |
|---|---|---|
| 隔离粒度 | 会话级 | 项目当前没有用户身份/认证，`session_id` 是唯一贯穿 API、LangGraph、记忆和数据库的稳定边界 |
| 字段命名 | 直接使用 `session_id` | 比抽象的 `scope_id` 更明确，且可直接关联 `sessions(id)` |
| 历史缓存 | 迁移时全部清空 | 旧数据没有归属信息，缓存本身可重建，保留只会引入歧义和泄漏风险 |
| 相似度查询 | session 内精确向量排序 | 每个 session 的 7 天缓存量很小；精确查询行为确定，不受全局 HNSW 过滤假阴性影响 |
| 全局 HNSW | 删除 | HNSW 先做全局近似候选再过滤 session，可能漏掉本 session 的真实最近邻 |
| 应用接口 | `session_id` 显式必传 | 让漏传在开发和测试阶段暴露，不保留隐式全局兼容路径 |
| 缺失处理 | fail-closed：跳过缓存 | 漏传最多损失一次缓存收益，不能重新引入跨会话返回 |
| 测试策略 | 自动单测 + 一次真实数据库手动验收 | 日常回归保持轻量，首次完成时验证真实约束、级联和隔离行为 |

### 3. 当前问题定位

当前实现存在四个直接缺口：

1. `semantic_cache` 表没有 session 字段，所有答案共用一个向量空间。
2. `SemanticCache.lookup()` 虽接收可选 `session_id`，但它只用于调用统计，查询 SQL 没有使用。
3. `SemanticCache.store()` 不接收 `session_id`，近重复检查和插入均为全局行为。
4. `cache_store` 节点没有把 state 中已有的 `session_id` 传给缓存层。

现有危险路径为：

```text
session A 生成答案并写入全局缓存
             ↓
session B 提出语义相近的问题
             ↓
全表向量最近邻命中 A 的记录
             ↓
B 收到 A 的答案与引用元数据
```

目标路径为：

```text
LangGraph state.session_id
          ↓ 显式必传
SemanticCache.lookup/store(session_id=...)
          ↓ SQL 强制过滤
semantic_cache.session_id = 当前 session
```

### 4. 数据库迁移

#### 4.1 新增迁移

新增文件：

```text
alembic/versions/0014_semantic_cache_session_scope.py
```

迁移头：

```python
revision = "0014"
down_revision = "0013"
```

#### 4.2 upgrade 顺序

升级操作必须按以下顺序执行：

1. 清空旧的全局缓存。
2. 删除原全局 HNSW 索引 `idx_semantic_cache_embedding`。
3. 新增 `session_id UUID NOT NULL`。
4. 新增指向 `sessions(id)` 的外键，设置 `ON DELETE CASCADE`。
5. 新建 `(session_id, created_at DESC)` B-tree 索引。

建议 SQL：

```sql
DELETE FROM semantic_cache;

DROP INDEX IF EXISTS idx_semantic_cache_embedding;

ALTER TABLE semantic_cache
    ADD COLUMN session_id UUID NOT NULL;

ALTER TABLE semantic_cache
    ADD CONSTRAINT fk_semantic_cache_session
    FOREIGN KEY (session_id)
    REFERENCES sessions(id)
    ON DELETE CASCADE;

CREATE INDEX idx_semantic_cache_session_created_at
    ON semantic_cache (session_id, created_at DESC);
```

这里不为 `session_id` 设置默认值。任何没有明确 session 的写入都应该失败，而不是进入一个共享的默认空间。

#### 4.3 downgrade 顺序

降级无法恢复已经清空的历史缓存，这是可接受的，因为缓存属于可重建数据。降级时应先再次清空当前 scoped 缓存，避免去掉 `session_id` 后把多个 session 的答案重新混为全局数据：

```sql
DELETE FROM semantic_cache;

DROP INDEX IF EXISTS idx_semantic_cache_session_created_at;

ALTER TABLE semantic_cache
    DROP CONSTRAINT IF EXISTS fk_semantic_cache_session;

ALTER TABLE semantic_cache
    DROP COLUMN IF EXISTS session_id;

CREATE INDEX idx_semantic_cache_embedding
    ON semantic_cache USING hnsw (embedding vector_cosine_ops);
```

#### 4.4 索引选择说明

查询条件为：

```sql
WHERE session_id = :session_id
  AND created_at > now() - TTL
ORDER BY embedding <=> :query_vector
LIMIT 1
```

`(session_id, created_at DESC)` 先把候选缩小到当前 session 的有效期内记录，再对这个小集合做精确余弦排序。相比全局 HNSW：

- 不会跨 session 返回数据；
- 不会因 ANN 的过滤顺序产生本 session 假未命中；
- 不依赖当前 `paradedb/paradedb:latest` 镜像中 pgvector 的具体迭代扫描版本；
- 性能由单 session、单 TTL 窗口内的数据量决定，而不是全表行数。

如果未来单个 session 在 TTL 内达到数万条缓存，再基于 P2-2 的实际数据重新评估 ANN；当前不提前引入分区或 iterative scan。

### 5. 缓存接口修改

#### 5.1 Protocol

修改 `rag/agent/type.py` 中的 `SemanticCacheProtocol`：

```python
class SemanticCacheProtocol(Protocol):
    async def lookup(
        self, query: str, *, session_id: str
    ) -> dict | None: ...

    async def store(
        self,
        query: str,
        answer: str,
        citations: list,
        *,
        session_id: str,
    ) -> None: ...
```

约束：

- `session_id` 使用 keyword-only 参数，调用处能直接看出隔离边界。
- 不允许 `None`，不提供默认值。
- 不新增 `scope_type`、默认 scope 或全局 fallback。

#### 5.2 SemanticCache.lookup

修改 `rag/agent/cache/manager.py`：

```python
async def lookup(self, query: str, *, session_id: str) -> dict | None:
```

查询 SQL 修改为：

```sql
SELECT id, answer, citations,
       1 - (embedding <=> %(vec)s) AS similarity
FROM semantic_cache
WHERE session_id = %(session_id)s
  AND created_at > now() - make_interval(hours => %(ttl)s)
ORDER BY embedding <=> %(vec)s
LIMIT 1
```

参数必须同时包含：

```python
{
    "session_id": session_id,
    "vec": vec,
    "ttl": self._ttl_hours,
}
```

命中后的计数更新增加 session 条件，防止未来代码错误地把其他 session 的行 ID 带入更新：

```sql
UPDATE semantic_cache
SET hit_count = hit_count + 1
WHERE id = %(id)s
  AND session_id = %(session_id)s
```

调用统计仍把该值写入 `CallRecord.session_id`，现有可观测语义不变。

#### 5.3 SemanticCache.store

签名修改为：

```python
async def store(
    self,
    query: str,
    answer: str,
    citations: list,
    *,
    session_id: str,
) -> None:
```

写入过程：

1. 计算 query embedding。
2. 使用同一个 `session_id` 调用 session 内最近邻查询。
3. 仅当当前 session 已有相似度达到阈值的记录时跳过插入。
4. 不同 session 的相同问题允许分别写入。

插入 SQL：

```sql
INSERT INTO semantic_cache
    (session_id, question, answer, citations, embedding)
VALUES
    (%(session_id)s, %(question)s, %(answer)s, %(citations)s, %(vec)s)
```

现有并发查重竞态保持原状：同一 session 的两个并发 miss 可能各插一条近重复记录。该竞态只导致少量冗余，不造成跨 session 泄漏，由 lookup 最近邻、TTL 和入库清空兜底，不在 P1-1 扩大范围处理。

#### 5.4 异常策略

缓存仍为 best-effort：

- lookup 的 embedding/数据库异常返回 miss；
- store 异常只记录 warning；
- 缓存故障不阻断主回答链路。

但“缺少 session”不在缓存管理器内兼容。管理器方法要求调用方显式传值；节点层负责 fail-closed。

### 6. LangGraph 节点传播

#### 6.1 cache_lookup

修改 `rag/agent/nodes/cache_lookup/lookup.py`：

```python
session_id = state.get("session_id")
if not session_id:
    logger.warning("语义缓存跳过：缺少 session_id")
    return state

hit = await cache.lookup(query, session_id=session_id)
```

顺序要求：先检查开关和 cache 注入，再检查 `session_id`，最后查询。缺失 session 时不得调用 embedding，也不得访问缓存表。

#### 6.2 cache_store

修改 `rag/agent/nodes/cache_store/store.py`：

```python
session_id = state.get("session_id")
if not session_id:
    logger.warning("语义缓存回写跳过：缺少 session_id")
    return state

await cache.store(
    query,
    answer,
    state.get("citations") or [],
    session_id=session_id,
)
```

空答案仍按现状跳过。建议检查顺序为：

1. 功能开关/cache 是否可用；
2. answer 是否非空；
3. `session_id` 是否非空；
4. 执行回写。

#### 6.3 请求生命周期保证

正常 HTTP 链路中，`rag/api/modules/chat/service.py` 会在运行图之前调用 `ensure_session()`，因此缓存写入时外键目标已经存在。

对于以后新增的非 HTTP 调用方：

- 必须先创建 session，再注入语义缓存；或
- 不注入语义缓存，让本次运行自然绕过缓存。

不允许为了兼容非 HTTP 调用而删除外键或使用虚拟默认 session。

### 7. 缓存管理操作保持不变

以下行为不改：

#### 7.1 文档入库失效

`clear_semantic_cache(pool)` 继续执行：

```sql
DELETE FROM semantic_cache
```

原因：共享知识库更新后，任意 session 的旧答案都可能引用已变化内容，无法安全地只清某一个 session。

#### 7.2 TTL 清理

`purge_expired(pool, ttl_hours)` 继续跨 session 删除过期记录。TTL 是记录生命周期，不是隔离边界。

#### 7.3 会话删除

无需在 Python 中额外删除缓存。`sessions` 行删除时由外键 `ON DELETE CASCADE` 原子清理该 session 的缓存，避免应用层两次删除之间出现残留。

### 8. 自动测试

#### 8.1 `tests/test_semantic_cache.py`

更新 fake cursor 和现有用例，并新增以下断言：

1. `lookup(query, session_id="...")` 的最近邻 SQL 含 `session_id` 条件。
2. SQL 参数包含调用方传入的 session ID。
3. 命中计数 UPDATE 同时包含 `id` 与 `session_id`。
4. `store(..., session_id="...")` 的查重查询限定同一 session。
5. INSERT 显式写入 `session_id`。
6. 相同 query 分别以 session A、B 写入时，两次调用参数保持各自 session，不进行全局查重。
7. 调用 `lookup()` 或 `store()` 时不传 session，因必填 keyword-only 参数直接触发 `TypeError`。
8. 数据库异常仍按现有策略吞掉，不改变主链路容错。

测试不能只断言 SQL 中“出现过 session_id”，还要断言绑定参数值正确，防止固定值或错传 state。

#### 8.2 `tests/test_nodes.py`

更新 `_FakeCache` 接口与调用记录：

```python
async def lookup(self, query, *, session_id): ...
async def store(self, query, answer, citations, *, session_id): ...
```

覆盖：

1. lookup 使用 rewrite query，并传入 state 的 `session_id`。
2. store 使用 rewrite query、答案、引用和同一个 `session_id`。
3. 两个 state 使用不同 session 时，fake cache 收到不同 session。
4. session 缺失时 lookup 不调用缓存。
5. session 为空字符串时 lookup 不调用缓存。
6. session 缺失/为空时 store 不调用缓存。
7. fail-closed 路径不设置 `cache_hit`，主图继续走正常检索生成。

#### 8.3 `tests/test_migration.py`

现有集成 smoke test 仍会在显式运行 `-m integration` 时执行，必须更新旧的 HNSW 断言，否则迁移完成后测试必然失败。最小更新为：

- `semantic_cache.session_id` 字段存在；
- `idx_semantic_cache_session_created_at` 存在；
- `idx_semantic_cache_embedding` 不存在。

P1-1 的完整跨 session 行为不新增到默认集成套件，按已确认策略在首次实现完成后手动验收。

### 9. 真实数据库手动验收

实现完成后执行一次以下验收，并在提交说明中记录结果。

#### 9.1 迁移结构

1. 启动 PostgreSQL。
2. 执行 `alembic upgrade head`。
3. 确认 `semantic_cache.session_id` 类型为 UUID 且不可为空。
4. 确认外键指向 `sessions(id)` 并带 `ON DELETE CASCADE`。
5. 确认 session/created_at B-tree 存在、旧 HNSW 不存在。
6. 确认迁移后旧缓存行数为 0。

#### 9.2 行为验收

建议使用前端或 `/chat/` 接口创建两个真实会话：

1. 开启 `SEMANTIC_CACHE_ENABLED=true`。
2. session A 首次提问，确认正常生成并写入缓存。
3. session A 再次提出相同或高度相似的问题，确认出现“缓存命中”。
4. session B 提出完全相同的问题，确认不出现“缓存命中”，而是重新检索和生成。
5. session B 再次提问，确认 B 能命中自己的缓存。
6. 查询数据库，确认 A、B 各自拥有记录，session ID 不混用。
7. 删除 session A，确认 A 的缓存行消失、B 的缓存仍存在。
8. 重启 API 后重复 A/B 查询，确认隔离不依赖进程内状态。

#### 9.3 降级验收

临时制造缓存不可用或传入无缓存的运行上下文，确认回答仍能通过正常检索/生成链路完成。缓存异常不得成为用户可见错误。

### 10. 文档同步

实现时同步修改：

1. `rag/config.py`：把“全局作用域”注释改为“session 作用域”。
2. `README.md`：把 `cache_lookup` 的“全局范围”说明改为“按 session 隔离”。
3. `README.md` 的语义缓存能力说明补充：精确相似度匹配、TTL、入库全局失效、会话删除级联。
4. `improvements.md` 不改需求原文；完成状态由提交记录或后续 checklist 管理。

无需修改：

- 前端请求结构：已有 `session_id`；
- `/chat/` API schema：已有 `session_id`；
- LangGraph 节点数量与连线；
- `.env`/`.env.example`：不新增配置项；
- 缓存相似度阈值与 TTL 默认值。

### 11. 文件改动清单

| 文件 | 改动 |
|---|---|
| `alembic/versions/0014_semantic_cache_session_scope.py` | 清空旧缓存；新增 session 外键；替换索引；提供安全 downgrade |
| `rag/agent/type.py` | 缓存 Protocol 的 lookup/store 改为必传 keyword-only `session_id` |
| `rag/agent/cache/manager.py` | 查找、查重、插入和命中计数全部按 session 限定 |
| `rag/agent/nodes/cache_lookup/lookup.py` | 从 state 显式传 session；缺失时 fail-closed |
| `rag/agent/nodes/cache_store/store.py` | 从 state 显式传 session；缺失时 fail-closed |
| `tests/test_semantic_cache.py` | 更新接口并覆盖 SQL/参数/查重隔离 |
| `tests/test_nodes.py` | 覆盖节点传播与缺失 session 跳过 |
| `tests/test_migration.py` | 更新已有 migration smoke 断言 |
| `rag/config.py` | 更新缓存作用域注释 |
| `README.md` | 更新架构和配置说明 |

### 12. 推荐实施顺序

1. 新增 0014 migration。
2. 修改 `SemanticCacheProtocol`，让所有漏传调用先在测试/类型层暴露。
3. 修改 `SemanticCache` SQL 与方法签名。
4. 修改 cache lookup/store 两个节点。
5. 更新和补充单元测试。
6. 更新已有 migration smoke test。
7. 更新 README 和配置注释。
8. 运行默认单元测试。
9. 执行一次真实数据库迁移与双 session 手动验收。
10. 检查 git diff，确认没有引入前端/API/工作流拓扑的非必要改动。

### 13. 验收标准

P1-1 只有同时满足以下条件才算完成：

- [ ] 数据库不允许无 `session_id` 的缓存记录。
- [ ] 所有最近邻查找都包含 `session_id = 当前会话`。
- [ ] 所有近重复检查都只在当前会话内执行。
- [ ] 所有写入都保存当前 `session_id`。
- [ ] 命中计数更新同时限定记录 ID 和 session ID。
- [ ] session 缺失时缓存节点跳过，不存在全局 fallback。
- [ ] 同一 session 重复问题能命中。
- [ ] 不同 session 的相同问题不能互相命中。
- [ ] 不同 session 可以分别缓存相同问题。
- [ ] 删除一个 session 只级联删除该 session 的缓存。
- [ ] 文档入库仍会清空全部 session 的旧缓存。
- [ ] 缓存故障仍降级为正常检索/生成，不拖垮回答。
- [ ] 默认单元测试通过。
- [ ] 真实 PostgreSQL 手动验收通过并记录结果。

### 14. 明确不在 P1-1 内处理

1. 用户认证、用户级 ACL、session 所有权校验。
2. `/sessions` 列表本身的多用户隔离。
3. 跨 session 共享公共问答缓存。
4. 同 session 并发写入导致的近重复竞态。
5. 缓存命中率与相似度分布观测（属于 P2-2）。
6. 单 session 达到大规模数据后的 ANN 优化。

因此，P1-1 修复的是“语义缓存跨会话返回答案”这一条明确的隐私缺陷，并不宣称系统已经具备完整的多租户安全能力。
