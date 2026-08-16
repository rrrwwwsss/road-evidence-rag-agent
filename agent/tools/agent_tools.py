from langchain_core.tools import tool
from rag.rag_service import RagSummarizeService
from rag.sql_service import SqlQueryService
from rag.vision_service import VisionService

rag = RagSummarizeService()
sql_service = SqlQueryService()
vision_service = VisionService()


@tool(description="从知识库检索参考资料（违法案例、判定标准、法规依据等），返回资料总结；用于回答历史案例相似度、道路养护/违法行为判定标准等需要知识库佐证的问题。doc_type 可选：查判定标准/法规传 standard，查历史案例传 case，不区分则传空字符串")
def rag_summarize(query: str, doc_type: str = "") -> str:
    return rag.rag_summarize(query, doc_type=doc_type)


@tool(description="根据自然语言查询问题查询案件数据库(SQLite)，返回查询结果；用于计数、统计、按时间/支队/地点/违法类型等条件的结构化查询")
def sql_query(question: str) -> str:
    return sql_service.query(question)


@tool(description="调用视觉大模型分析图片中的违法行为；入参image为图片ID（用户消息中的img_xxx）或图片URL，question为要分析的问题，context为可选的RAG判定标准/参考案例")
def vision_analyze(image: str, question: str, context: str = "") -> str:
    return vision_service.analyze(image, question, context)


@tool(description="无入参，无返回值，调用后触发中间件自动为报告生成的场景动态注入上下文信息，为后续提示词切换提供上下文信息")
def fill_context_for_report():
    return "fill_context_for_report已调用"