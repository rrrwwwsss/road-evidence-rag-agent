"""规则优先、模型可选兜底的意图分类器。"""
from __future__ import annotations

import re
from typing import Any

from agent.intents.models import IntentDecision, IntentGoal, IntentType, QueryEntities
from utils.logger_handler import logger


REPORT_WORDS = ("报告", "报表", "执法文书", "取证文书")
STAT_WORDS = ("多少", "几条", "统计", "汇总", "数量", "排名", "最多", "最少", "明细", "查询案件")
KNOWLEDGE_WORDS = ("标准", "依据", "法规", "如何判断", "怎么判断", "认定", "相似案例", "历史案例", "典型案例", "区别")
IMAGE_WORDS = ("图片", "图中", "这张图", "照片", "识别")
IMAGE_CASE_WORDS = (
    "类似案例", "相似案例", "类似的案例", "相似的案例", "历史案例",
    "违法案例", "查找案例", "找案例", "有没有案例", "是否有案例",
)


class IntentClassifier:
    def __init__(self, fallback_model: Any | None = None, fallback_threshold: float = 0.7):
        self.fallback_model = fallback_model
        self.fallback_threshold = fallback_threshold

    def classify(self, query: str) -> IntentDecision:
        entities = self.extract_entities(query)
        has_report = any(word in query for word in REPORT_WORDS)
        has_image = bool(entities.image_ids) or any(word in query for word in IMAGE_WORDS)
        has_stats = any(word in query for word in STAT_WORDS)
        has_knowledge = any(word in query for word in KNOWLEDGE_WORDS)
        has_image_case_search = any(word in query for word in IMAGE_CASE_WORDS)
        # “有没有违法案例”是在找案例；只有明确指向当前图片/行为时才属于违法判定。
        has_image_judgement = bool(re.search(
            r"(?:这张图|这张图片|图中|当前图片|当前行为|这个行为|该行为).{0,12}"
            r"(?:是否违法(?!案例)|有没有违法(?!案例)|是不是违法(?!案例)|是否属于|判定|认定)",
            query,
        )) or bool(re.search(r"(?:判断|判定|认定).{0,12}(?:这张图|图中|当前图片|当前行为|该行为)", query))
        has_current_image_reference = any(
            word in query for word in ("这张图", "这张图片", "这个图片", "当前图片", "图中")
        )
        has_image_judgement = has_image_judgement or bool(
            has_current_image_reference
            and re.search(r"(?:是否|有没有|是不是|判断|判定|认定|分析).{0,20}违法(?:行为)?(?!案例)", query)
        )

        secondary: list[IntentType] = []
        goals: list[IntentGoal] = []
        if has_report:
            primary, confidence, reason = IntentType.REPORT_GENERATION, 0.98, "命中报告生成关键词"
            secondary = [IntentType.CASE_QUERY, IntentType.KNOWLEDGE_QUERY]
            # 报告 Skill 内部已经包含数据库查询和证据检索，不拆成重复执行的三个目标。
            goals = [IntentGoal(intent=primary, objective="生成有数据和证据支撑的报告")]
        elif has_image and has_image_case_search and has_image_judgement:
            primary, confidence, reason = IntentType.IMAGE_ASSESSMENT, 0.98, "检测到图片案例检索与违法判定两个目标"
            secondary = [IntentType.IMAGE_CASE_SEARCH]
            goals = [
                IntentGoal(intent=IntentType.IMAGE_CASE_SEARCH, objective="查找与当前图片相似的历史案例"),
                IntentGoal(
                    intent=IntentType.IMAGE_ASSESSMENT,
                    objective="结合历史案例和判定标准评估当前图片",
                    depends_on=[IntentType.IMAGE_CASE_SEARCH],
                ),
            ]
        elif has_image and has_image_case_search and not has_image_judgement:
            primary, confidence, reason = IntentType.IMAGE_CASE_SEARCH, 0.97, "检测到图片相似案例检索请求"
            secondary = [IntentType.KNOWLEDGE_QUERY]
        elif has_image:
            primary, confidence, reason = IntentType.IMAGE_ASSESSMENT, 0.96, "检测到图片或图片分析表达"
            if has_knowledge:
                secondary.append(IntentType.KNOWLEDGE_QUERY)
                goals = [
                    IntentGoal(intent=IntentType.KNOWLEDGE_QUERY, objective="检索用户要求的判定依据"),
                    IntentGoal(
                        intent=IntentType.IMAGE_ASSESSMENT,
                        objective="结合判定依据评估当前图片",
                        depends_on=[IntentType.KNOWLEDGE_QUERY],
                    ),
                ]
        elif has_stats:
            primary, confidence, reason = IntentType.CASE_QUERY, 0.92, "命中案件统计或明细查询表达"
            if has_knowledge:
                secondary.append(IntentType.KNOWLEDGE_QUERY)
                goals = [
                    IntentGoal(intent=IntentType.CASE_QUERY, objective="查询案件数据"),
                    IntentGoal(intent=IntentType.KNOWLEDGE_QUERY, objective="检索相关标准或案例"),
                ]
        elif has_knowledge:
            primary, confidence, reason = IntentType.KNOWLEDGE_QUERY, 0.9, "命中标准、依据或案例检索表达"
        elif query.strip():
            primary, confidence, reason = IntentType.GENERAL_CHAT, 0.62, "未命中业务规则，按一般问答处理"
        else:
            primary, confidence, reason = IntentType.UNKNOWN, 1.0, "输入为空"

        if not goals:
            goals = [IntentGoal(intent=primary, objective=query.strip())]

        decision = IntentDecision(
            primary_intent=primary,
            secondary_intents=secondary,
            goals=goals,
            entities=entities,
            confidence=confidence,
            needs_clarification=primary == IntentType.UNKNOWN,
            clarification_question="请补充需要查询、判定或生成报告的具体内容。" if primary == IntentType.UNKNOWN else None,
            reason=reason,
        )
        if decision.confidence < self.fallback_threshold and self.fallback_model is not None:
            return self._model_fallback(query, decision)
        return decision

    def _model_fallback(self, query: str, default: IntentDecision) -> IntentDecision:
        prompt = (
            "请识别用户请求中的一个或多个独立业务目标，分类为 case_query、knowledge_query、"
            "image_case_search、image_assessment、report_generation、general_chat 或 unknown，并提取实体。"
            "同时填写 goals 及目标依赖；查找图片相似案例后再判断图片时必须拆成两个 goals。"
            "只根据用户原文判断。\n"
            f"用户请求：{query}"
        )
        try:
            structured = self.fallback_model.with_structured_output(IntentDecision)
            result = structured.invoke(prompt)
            result.reason = result.reason or "模型兜底分类"
            return result
        except Exception as exc:
            logger.warning("[IntentClassifier]模型兜底失败，使用规则结果：%s", exc)
            return default

    @staticmethod
    def extract_entities(query: str) -> QueryEntities:
        dates = re.findall(r"\d{4}(?:年|-)\d{1,2}(?:(?:月|-)\d{1,2}日?)?", query)
        image_ids = re.findall(r"img_[A-Za-z0-9]+", query)
        case_ids = re.findall(r"(?:案例|案件)(?:编号)?[：:\s]*([A-Za-z0-9_-]+)", query)
        raw_units = re.findall(r"([\u4e00-\u9fff]{1,12}(?:支队|执法队|大队))", query)
        units = []
        for unit in raw_units:
            unit = re.sub(r"^.*?[年月日]", "", unit)
            unit = re.sub(r"^(?:请|统计|查询|生成|为|的)+", "", unit)
            if unit:
                units.append(unit)
        violation_types = [
            word for word in (
                "擅自占用公路", "擅自挖掘公路", "摆摊设点", "堆放物品",
                "移动井盖", "设置非公路标志", "道路养护",
            ) if word in query
        ]
        time_range = dates[0] if dates else None
        return QueryEntities(
            dates=dates,
            time_range=time_range,
            units=list(dict.fromkeys(units)),
            violation_types=violation_types,
            case_ids=case_ids,
            image_ids=list(dict.fromkeys(image_ids)),
        )
