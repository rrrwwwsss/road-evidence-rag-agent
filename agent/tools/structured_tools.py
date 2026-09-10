"""对现有 RAG、SQL、Vision 服务的结构化 Tool 适配器。"""
from __future__ import annotations

from time import perf_counter
from typing import Any

from agent.tools.contracts import EvidenceSource, ToolResult
from utils.logger_handler import logger


def _elapsed_ms(start: float) -> int:
    return round((perf_counter() - start) * 1000)


class RagTool:
    name = "retrieve_knowledge"

    def __init__(self, service: Any):
        self.service = service

    def invoke(self, query: str, doc_type: str = "", instructions: str = "") -> ToolResult:
        started = perf_counter()
        try:
            docs = self.service.retriever_docs(query, doc_type=doc_type)
            context_parts: list[str] = []
            sources: list[EvidenceSource] = []
            for index, doc in enumerate(docs, 1):
                metadata = dict(doc.metadata or {})
                source_id = str(metadata.get("source") or metadata.get("document_name") or index)
                context_parts.append(
                    f"【参考资料{index}】{doc.page_content}\n元数据：{metadata}"
                )
                sources.append(EvidenceSource(
                    source_type="knowledge_base",
                    source_id=source_id,
                    title=str(metadata.get("title") or source_id),
                    excerpt=doc.page_content[:500],
                    metadata=metadata,
                ))

            if not docs:
                return ToolResult(
                    tool_name=self.name,
                    ok=False,
                    data={
                        "query": query,
                        "doc_type": doc_type,
                        "backend": getattr(self.service, "last_backend", "unknown"),
                        "upstream_error": getattr(self.service, "last_retrieval_error", None),
                    },
                    error_code="NO_RESULTS",
                    error_message=(getattr(self.service, "last_retrieval_error", None)
                                   or "知识库未检索到相关资料"),
                    latency_ms=_elapsed_ms(started),
                )

            context = "\n".join(context_parts)
            summary_input = query
            if instructions.strip():
                summary_input = f"{query}\n\n【当前 Skill 输出要求】\n{instructions}"
            summary = self.service.chain.invoke({"input": summary_input, "context": context})
            return ToolResult(
                tool_name=self.name,
                ok=True,
                data={
                    "query": query,
                    "doc_type": doc_type,
                    "backend": getattr(self.service, "last_backend", "unknown"),
                    "upstream_error": getattr(self.service, "last_retrieval_error", None),
                    "summary": summary,
                    "context": context,
                    "skill_prompt_applied": bool(instructions.strip()),
                },
                sources=sources,
                latency_ms=_elapsed_ms(started),
            )
        except Exception as exc:
            logger.error("[RagTool]调用失败：%s", exc, exc_info=True)
            result = ToolResult.failure(self.name, exc)
            result.latency_ms = _elapsed_ms(started)
            return result


class SqlTool:
    name = "query_cases"

    def __init__(self, service: Any):
        self.service = service

    def invoke(self, question: str) -> ToolResult:
        started = perf_counter()
        try:
            payload = self.service.query_structured(question)
            rows = payload.get("rows", [])
            source = EvidenceSource(
                source_type="database",
                source_id="results",
                title="案件数据库查询",
                excerpt=f"SQL: {payload.get('sql', '')}",
                metadata={"columns": payload.get("columns", []), "row_count": len(rows)},
            )
            return ToolResult(
                tool_name=self.name,
                ok=True,
                data=payload,
                sources=[source],
                latency_ms=_elapsed_ms(started),
                truncated=bool(payload.get("truncated")),
            )
        except Exception as exc:
            logger.error("[SqlTool]调用失败：%s", exc, exc_info=True)
            result = ToolResult.failure(self.name, exc)
            result.latency_ms = _elapsed_ms(started)
            return result


class VisionTool:
    name = "analyze_image"

    def __init__(self, service: Any):
        self.service = service

    def invoke(self, image: str, question: str, context: str = "",
               mode: str = "assessment", instructions: str = "") -> ToolResult:
        started = perf_counter()
        try:
            analysis = self.service.analyze(
                image=image,
                question=question,
                context=context,
                mode=mode,
                instructions=instructions,
            )
            missing = analysis.startswith("未找到图片") or analysis.startswith("视觉模型调用失败")
            return ToolResult(
                tool_name=self.name,
                ok=not missing,
                data={
                    "image_id": image,
                    "analysis": analysis,
                    "mode": mode,
                    "skill_prompt_applied": bool(instructions.strip()),
                },
                sources=[EvidenceSource(
                    source_type="image",
                    source_id=image,
                    title="用户上传图片",
                    excerpt=analysis[:500],
                )] if not missing else [],
                error_code="VISION_ERROR" if missing else None,
                error_message=analysis if missing else None,
                latency_ms=_elapsed_ms(started),
            )
        except Exception as exc:
            logger.error("[VisionTool]调用失败：%s", exc, exc_info=True)
            result = ToolResult.failure(self.name, exc)
            result.latency_ms = _elapsed_ms(started)
            return result


class ToolRegistry:
    """集中注册原子工具，Skill 不直接依赖底层服务。"""

    def __init__(self, rag: RagTool, sql: SqlTool, vision: VisionTool):
        self.rag = rag
        self.sql = sql
        self.vision = vision

    @classmethod
    def default(cls) -> "ToolRegistry":
        from rag.rag_service import RagSummarizeService
        from rag.sql_service import SqlQueryService
        from rag.vision_service import VisionService

        return cls(
            rag=RagTool(RagSummarizeService()),
            sql=SqlTool(SqlQueryService()),
            vision=VisionTool(VisionService()),
        )
