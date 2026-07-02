from __future__ import annotations

from collections import Counter

from rag.document.entity_extraction import ExtractionResult


def aggregate(
    chunks: list[tuple[str, ExtractionResult]],
) -> tuple[list[dict], list[dict]]:
    """把整篇文档各 chunk 的抽取结果预聚合为待写入 Neo4j 的实体/关系。

    - 实体按 name 合并：chunk_ids 取并集；type 取众数（平票取首次出现）。
    - 关系按规范化方向 (min, max) 合并：keywords 取并集。
    - 丢弃两端不在实体集合内的关系。
    每项入参为 (chunk_uid, ExtractionResult)。
    """
    ent_chunks: dict[str, set[str]] = {}
    ent_types: dict[str, list[str]] = {}
    for chunk_uid, result in chunks:
        for ent in result.entities:
            name = ent.name
            ent_chunks.setdefault(name, set()).add(chunk_uid)
            ent_types.setdefault(name, []).append(ent.type or "其他")

    entities: list[dict] = []
    for name, chunk_ids in ent_chunks.items():
        # Counter.most_common 对平票保留首次插入顺序 → 众数、平票取首现
        top_type = Counter(ent_types[name]).most_common(1)[0][0]
        entities.append(
            {"name": name, "type": top_type, "chunk_ids": sorted(chunk_ids)}
        )

    valid = set(ent_chunks)
    rel_keywords: dict[tuple[str, str], set[str]] = {}
    for _chunk_uid, result in chunks:
        for rel in result.relationships:
            s, t = rel.source, rel.target
            if s == t or s not in valid or t not in valid:
                continue
            key = (s, t) if s <= t else (t, s)
            kws = rel_keywords.setdefault(key, set())
            if rel.keywords:
                for kw in str(rel.keywords).split(","):
                    kw = kw.strip()
                    if kw:
                        kws.add(kw)

    relations = [
        {"source": s, "target": t, "keywords": sorted(kws)}
        for (s, t), kws in rel_keywords.items()
    ]
    return entities, relations
