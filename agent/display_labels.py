"""内部 Intent/Skill 标识与用户界面中文名称的映射。"""
from __future__ import annotations

from agent.intents.models import IntentType


INTENT_LABELS: dict[IntentType, str] = {
    IntentType.CASE_QUERY: "案件统计与明细查询",
    IntentType.KNOWLEDGE_QUERY: "判定标准与历史案例检索",
    IntentType.IMAGE_CASE_SEARCH: "图片相似案例检索",
    IntentType.IMAGE_ASSESSMENT: "图片违法与养护判定",
    IntentType.REPORT_GENERATION: "执法与取证报告生成",
    IntentType.GENERAL_CHAT: "一般问答",
    IntentType.UNKNOWN: "未识别意图",
}

SKILL_LABELS: dict[str, str] = {
    "case_query": "案件统计与明细查询",
    "knowledge_query": "判定标准与历史案例检索",
    "image_case_search": "图片相似案例检索",
    "image_assessment": "图片违法与养护判定",
    "report_generation": "执法与取证报告生成",
}


def intent_label(intent: IntentType) -> str:
    return INTENT_LABELS.get(intent, intent.value)


def skill_label(skill_name: str) -> str:
    return SKILL_LABELS.get(skill_name, skill_name)


def localize_trace(text: str) -> str:
    """兼容已经保存在 Streamlit 会话中的旧英文轨迹。"""
    localized = text
    for internal_name, label in sorted(SKILL_LABELS.items(), key=lambda item: len(item[0]), reverse=True):
        localized = localized.replace(internal_name, label)
    return localized
