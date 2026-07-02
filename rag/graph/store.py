from __future__ import annotations

from neo4j import AsyncDriver

# 纯 Cypher 列表去重(neo4j:5-community 无 APOC):[x IN old WHERE NOT x IN new] + new
_WRITE_ENTITIES = """
UNWIND $entities AS ent
MERGE (e:Entity {name: ent.name})
ON CREATE SET e.type = ent.type
SET e.chunk_ids = [x IN coalesce(e.chunk_ids, []) WHERE NOT x IN ent.chunk_ids] + ent.chunk_ids,
    e.doc_ids   = [x IN coalesce(e.doc_ids, []) WHERE x <> $doc] + [$doc]
"""

_WRITE_RELATIONS = """
UNWIND $relations AS rel
MATCH (a:Entity {name: rel.source})
MATCH (b:Entity {name: rel.target})
MERGE (a)-[r:RELATES]-(b)
SET r.keywords = [x IN coalesce(r.keywords, []) WHERE NOT x IN rel.keywords] + rel.keywords,
    r.doc_ids  = [x IN coalesce(r.doc_ids, []) WHERE x <> $doc] + [$doc]
"""

_PURGE_EDGES = """
MATCH ()-[r:RELATES]->()
WHERE $doc IN r.doc_ids
SET r.doc_ids = [d IN r.doc_ids WHERE d <> $doc]
WITH r WHERE size(r.doc_ids) = 0
DELETE r
"""

_PURGE_NODES = """
MATCH (e:Entity)
WHERE $doc IN e.doc_ids
SET e.chunk_ids = [c IN e.chunk_ids WHERE NOT c STARTS WITH $prefix],
    e.doc_ids   = [d IN e.doc_ids WHERE d <> $doc]
WITH e WHERE size(e.doc_ids) = 0
DETACH DELETE e
"""


async def purge_document(driver: AsyncDriver, database: str, document_id: str) -> None:
    """重跑前清理该文档在图中的旧贡献(先边后节点,一个事务完成)。"""
    prefix = f"{document_id}:"

    async def _tx(tx):
        await tx.run(_PURGE_EDGES, doc=document_id)
        await tx.run(_PURGE_NODES, doc=document_id, prefix=prefix)

    async with driver.session(database=database) as session:
        await session.execute_write(_tx)


async def write_graph(
    driver: AsyncDriver,
    database: str,
    document_id: str,
    entities: list[dict],
    relations: list[dict],
) -> None:
    """批量 MERGE 写入实体与关系(实体先于关系;一个事务完成)。"""
    if not entities:
        return

    async def _tx(tx):
        await tx.run(_WRITE_ENTITIES, entities=entities, doc=document_id)
        if relations:
            await tx.run(_WRITE_RELATIONS, relations=relations, doc=document_id)

    async with driver.session(database=database) as session:
        await session.execute_write(_tx)
