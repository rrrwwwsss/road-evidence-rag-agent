"""执法辅助场景中的证据完整性策略。"""
from __future__ import annotations

from agent.tools.contracts import ToolResult


JUDGEMENT_WORDS = ("是否", "违法", "养护", "施工", "判定", "认定", "属于")


def requires_reference_evidence(question: str) -> bool:
    return any(word in question for word in JUDGEMENT_WORDS)


def has_unconfirmed_case(result: ToolResult) -> bool:
    """识别知识库上下文中明确的未确认违法记录。"""
    if not result.ok:
        return False
    text = str(result.data)
    return "is_committed=0" in text or "is_committed：0" in text or "未确认违法" in text
