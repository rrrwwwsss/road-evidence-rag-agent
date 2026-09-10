"""把内部进度转换为可展示、可持久化的审计事件。"""
from __future__ import annotations

from datetime import datetime
import re
from typing import Any

from agent.tools.contracts import ToolResult


EVENT_EXPLANATIONS: dict[str, str] = {
    "request_received": "已为本次请求创建执行状态，后续步骤都会记录在同一条审计轨迹中。",
    "memory_loaded": "已从当前会话中加载限定窗口内的用户和助手消息，供本轮上下文使用。",
    "intent_started": "正在从用户原文中识别业务目标和查询条件。",
    "intent_classified": "已完成业务意图分类；这是路由结果，不是模型内部思维链。",
    "reasoning_summary": "以下是系统根据结构化状态生成的可审计推理摘要，可核对分类和规划依据。",
    "analysis_summary": "以下汇总结论使用的业务能力、证据、警告和执行状态。",
    "clarification_requested": "当前信息不足，系统暂停执行并请求用户补充必要条件。",
    "general_answer_generated": "该请求无需业务工具，已由一般问答能力完成。",
    "planning_started": "正在把一个或多个业务目标转换为带依赖关系的执行步骤。",
    "plan_created": "执行顺序已经确定；有依赖的步骤只有在前置步骤成功后才会运行。",
    "skill_selected": "已进入对应业务 Skill；Skill 会按固定规则调用允许的工具。",
    "tool_started": "正在调用业务工具。此处仅展示调用目的，不展示模型隐藏推理。",
    "tool_completed": "工具调用成功，返回结果将作为后续分析的可追溯依据。",
    "tool_failed": "工具调用未成功；失败信息会进入执行状态并影响后续依赖步骤。",
    "tool_observation": "以下是本次工具调用的脱敏参数、实际命令和结果预览，可用于核对执行依据。",
    "analysis_started": "正在根据结构化工具结果生成用户可读的事实总结。",
    "analysis_completed": "结构化结果汇总完成。",
    "model_started": "正在基于已经取得的事实和证据生成业务文本。",
    "model_completed": "业务文本生成完成。",
    "plan_step_completed": "当前计划步骤已完成，其共享产物可供后续步骤复用。",
    "plan_step_failed": "当前计划步骤失败，依赖它的后续步骤将不会继续执行。",
    "plan_step_skipped": "由于前置步骤未完成，为避免无依据输出，当前步骤已跳过。",
    "orchestration_completed": "计划中的全部任务均已执行完成。",
    "orchestration_incomplete": "至少一个计划步骤失败或被跳过，最终结果可能不完整。",
    "orchestration_failed": "调度过程中发生异常，系统已停止本次执行。",
}


def event_status(event_name: str) -> str:
    if event_name.endswith(("failed", "incomplete")) or event_name == "tool_failed":
        return "error"
    if event_name.endswith(("completed", "classified", "created")) or event_name in {
        "tool_completed", "general_answer_generated", "reasoning_summary", "analysis_summary",
        "memory_loaded",
    }:
        return "complete"
    if event_name.endswith("skipped"):
        return "skipped"
    return "running"


def event_category(event_name: str) -> str:
    if event_name in {"reasoning_summary", "analysis_summary"}:
        return "分析依据"
    if event_name.startswith("memory"):
        return "上下文记忆"
    if event_name.startswith("intent"):
        return "意图识别"
    if event_name.startswith("plan") or event_name.startswith("orchestration"):
        return "任务编排"
    if event_name.startswith("skill"):
        return "业务能力"
    if event_name.startswith("tool"):
        return "工具调用"
    if event_name.startswith("model") or event_name.startswith("analysis"):
        return "结果分析"
    return "系统处理"


def build_audit_event(
    event_name: str,
    detail: str,
    sequence: int,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """仅构造可公开的过程说明，禁止放入提示词、模型草稿或隐藏推理。"""
    payload = {
        "id": f"event_{sequence}",
        "sequence": sequence,
        "event": event_name,
        "title": detail,
        "category": event_category(event_name),
        "status": event_status(event_name),
        "explanation": EVENT_EXPLANATIONS.get(
            event_name,
            "这是系统执行过程中记录的一条可审计业务事件。",
        ),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    if details:
        payload["details"] = details
        if event_name == "tool_observation":
            payload["status"] = "complete" if details.get("ok") else "error"
    return payload


TOOL_LABELS = {
    "query_cases": "案件数据库查询",
    "retrieve_knowledge": "知识库检索",
    "analyze_image": "图片分析",
}

BACKEND_LABELS = {
    "dify": "Dify 知识库",
    "local_chroma": "本地 Chroma 向量库",
    "unavailable": "检索后端不可用",
    "unknown": "未知后端",
}


def _safe_text(value: Any, max_chars: int) -> str:
    text = str(value or "")
    text = re.sub(
        r"(?i)(authorization|api[_ -]?key)\s*[:=]\s*\S+",
        r"\1=[已隐藏]",
        text,
    )
    text = re.sub(r"\b(?:sk|dataset)-[A-Za-z0-9_-]{8,}\b", "[密钥已隐藏]", text)
    return text if len(text) <= max_chars else text[:max_chars] + "…（已截断）"


def build_tool_call_details(result: ToolResult) -> dict[str, Any]:
    """生成允许展示的 Tool 调用详情；明确排除提示词、请求头和完整 RAG 上下文。"""
    data = result.data if isinstance(result.data, dict) else {}
    details: dict[str, Any] = {
        "tool_name": result.tool_name,
        "tool_label": TOOL_LABELS.get(result.tool_name, result.tool_name),
        "ok": result.ok,
        "latency_ms": result.latency_ms,
        "truncated": result.truncated,
    }
    if not result.ok:
        details["error"] = _safe_text(result.error_message, 800)

    if result.tool_name == "query_cases":
        rows = data.get("rows", []) if isinstance(data.get("rows", []), list) else []
        details["command"] = _safe_text(data.get("sql", ""), 3000)
        details["result"] = {
            "columns": list(data.get("columns", []))[:50],
            "row_count": len(rows),
            "rows_preview": rows[:10],
            "preview_limit": 10,
        }
    elif result.tool_name == "retrieve_knowledge":
        details["request"] = {
            "query": _safe_text(data.get("query", ""), 500),
            "doc_type": data.get("doc_type") or "全部",
            "backend": BACKEND_LABELS.get(
                data.get("backend", "unknown"),
                data.get("backend", "unknown"),
            ),
        }
        details["result"] = {
            "source_count": len(result.sources),
            "summary": _safe_text(data.get("summary", ""), 2000),
            "upstream_error": _safe_text(data.get("upstream_error", ""), 800),
        }
    elif result.tool_name == "analyze_image":
        details["request"] = {
            "image_id": _safe_text(data.get("image_id", ""), 100),
            "mode": data.get("mode", "assessment"),
        }
        details["result"] = {
            "analysis": _safe_text(data.get("analysis", ""), 2000),
        }

    details["sources"] = [
        {
            "title": _safe_text(source.title or source.source_id, 200),
            "source_type": source.source_type,
            "excerpt": _safe_text(source.excerpt, 400),
        }
        for source in result.sources[:5]
    ]
    return details
