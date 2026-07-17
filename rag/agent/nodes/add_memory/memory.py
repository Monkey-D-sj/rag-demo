from langgraph.runtime import Runtime

from rag.agent.type import ContextSchema, MyState


async def add_memory(state: MyState, runtime: Runtime[ContextSchema]) -> MyState:
    """将本轮问答结果写回短期 + 长期记忆。

    经 generate → cache_store → add_memory 或缓存命中时 cache_lookup → add_memory 到达,
    记忆写入失败不阻断流程、不阻塞流式响应(generate/cache_lookup 已将 token 全部下发)。
    """
    memory_manager = runtime.context.memory_manager
    if memory_manager is None:
        return state

    await memory_manager.persist_turn(
        state["session_id"],
        state["raw_query"],
        state.get("generated", ""),
    )
    return state
