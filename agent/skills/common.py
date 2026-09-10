"""各 Skill handler 共用的小型确定性组件。"""
from __future__ import annotations

from typing import Callable

from agent.tools.contracts import ToolResult


ProgressCallback = Callable[[str, str], None] | None


def report_progress(callback: ProgressCallback, event: str, detail: str) -> None:
    if callback is not None:
        callback(event, detail)


def format_sql_result(result: ToolResult) -> str:
    if not result.ok:
        return f"案件数据库查询失败：{result.error_message}"
    payload = result.data
    rows = payload.get("rows", [])
    if not rows:
        return f"未查询到匹配案件。\n\n查询语句：`{payload.get('sql', '')}`"
    columns = payload.get("columns", [])
    lines = [f"查询到 {len(rows)} 条结果。", "", "| " + " | ".join(columns) + " |"]
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows[:50]:
        values = [str(row.get(column, "")).replace("|", "\\|") for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    if result.truncated or len(rows) > 50:
        lines.append("\n结果已截断，请缩小查询范围后查看完整明细。")
    lines.append(f"\n查询语句：`{payload.get('sql', '')}`")
    return "\n".join(lines)
