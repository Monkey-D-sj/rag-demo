from neo4j import AsyncDriver, AsyncGraphDatabase

from rag.config import Settings

# 单库去重键:实体名唯一。多库隔离需改为 (kb_id, name) 复合约束(见设计 §12)。
_ENTITY_CONSTRAINT = (
    "CREATE CONSTRAINT entity_key IF NOT EXISTS "
    "FOR (e:Entity) REQUIRE e.name IS UNIQUE"
)


def create_neo4j_driver(settings: Settings) -> AsyncDriver:
    """创建 Neo4j 异步 driver（原生 async，无需 to_thread）。"""
    return AsyncGraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
    )


async def ensure_graph_constraints(driver: AsyncDriver, database: str) -> None:
    """幂等建约束；worker 启动时调用一次。"""
    async with driver.session(database=database) as session:
        await session.run(_ENTITY_CONSTRAINT)
