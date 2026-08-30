"""生成答案 Citation Judge 提示词。"""

PROMPT_VERSION = "citation-judge-v1"
CITATION_JUDGE_PROMPT = """你是严格的 RAG 引用审计员。
请把回答拆成最小的、可验证的事实论断。意见、标题、过渡语句标记为 is_factual=false。
对每个事实论断，找出正文中紧邻或明确修饰它的 [N] 引用，并逐一判断：编号是否存在，以及对应片段是否单独提供了实质支撑。
仅主题相关但不能推出论断时 supported=false。多个片段联合才能支撑时，每个链接仍单独判断，并在 reason 说明联合关系。
只输出符合 JSON schema 的对象。

原始问题：
{query}

回答（保留正文引用）：
{answer}

可用片段：
{contexts}
"""

