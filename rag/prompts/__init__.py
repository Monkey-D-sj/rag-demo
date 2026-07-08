"""集中管理的提示词模块。

每个子模块对应一个业务域，提示词从业务代码中抽离至此，方便：
- 统一查找、审计、版本对比
- 非开发人员（PM/领域专家）直接编辑
- 后续接入 A/B 测试或远程配置
"""

from rag.prompts.generate import direct_system_prompt, system_prompt as generate_system_prompt
from rag.prompts.query import system_prompt as query_system_prompt
from rag.prompts.entity_extraction import (
    entity_extract_prompt,
    entity_types_guidance,
    human_extract_prompt,
)
from rag.prompts.eval import GOLDEN_GENERATION_PROMPT

__all__ = [
    "generate_system_prompt",
    "direct_system_prompt",
    "query_system_prompt",
    "entity_extract_prompt",
    "human_extract_prompt",
    "entity_types_guidance",
    "GOLDEN_GENERATION_PROMPT",
]
