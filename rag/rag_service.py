"""
总结服务类：用户提问，搜索参考资料，将提问和参考资料提交给模型，让模型总结回复
"""
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from rag.vector_store import get_vector_store_service
from rag.dify_retriever import DifyRetriever
from utils.prompt_loader import load_rag_prompts
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model
from utils.logger_handler import logger


def print_prompt(prompt):
    print("="*20)
    print(prompt.to_string())
    print("="*20)
    return prompt


class RagSummarizeService(object):
    def __init__(self):
        # Dify 知识库检索（未配置 DIFY_API_KEY / dataset_id 时自动回退本地向量库）
        self.dify_retriever = DifyRetriever()
        # Chroma 仅在 Dify 不可用/无结果时懒加载，避免 Streamlit 启动时重复创建 Rust 客户端。
        self._local_retriever = None
        self.last_backend = "none"
        self.last_retrieval_error: str | None = None
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self._init_chain()

    def _init_chain(self):
        chain = self.prompt_template | print_prompt | self.model | StrOutputParser()
        return chain

    def retriever_docs(self, query: str, doc_type: str = "") -> list[Document]:
        self.last_backend = "none"
        self.last_retrieval_error = None
        if self.dify_retriever.available:
            docs = self.dify_retriever.retrieve(query, doc_type=doc_type)
            if docs:
                self.last_backend = "dify"
                return docs
            dify_error = getattr(self.dify_retriever, "last_error", None)
            if isinstance(dify_error, str) and dify_error:
                self.last_retrieval_error = dify_error
                logger.warning("[rag_service]Dify检索失败（%s），回退本地向量库", dify_error)
            else:
                logger.warning("[rag_service]Dify检索成功但无命中，回退本地向量库")

        retriever = self._get_local_retriever()
        if retriever is None:
            self.last_backend = "unavailable"
            return []
        try:
            docs = retriever.invoke(query)
            self.last_backend = "local_chroma"
            return docs
        except Exception as exc:
            self.last_backend = "unavailable"
            self.last_retrieval_error = str(exc)
            logger.error("[rag_service]本地向量库检索失败：%s", exc, exc_info=True)
            return []

    def _get_local_retriever(self):
        if self._local_retriever is not None:
            return self._local_retriever
        try:
            self._local_retriever = get_vector_store_service().get_retriever()
            return self._local_retriever
        except Exception as exc:
            # 本地兜底不可用不应阻止 Streamlit 和 Dify 主路径启动。
            logger.error("[rag_service]本地向量库初始化失败，已禁用本次回退：%s", exc, exc_info=True)
            return None

    def rag_summarize(self, query: str, doc_type: str = "") -> str:

        context_docs = self.retriever_docs(query, doc_type=doc_type)

        context = ""
        counter = 0
        for doc in context_docs:
            counter += 1
            context += f"【参考资料{counter}】: 参考资料：{doc.page_content} | 参考元数据：{doc.metadata}\n"

        return self.chain.invoke(
            {
                "input": query,
                "context": context,
            }
        )


if __name__ == '__main__':
    rag = RagSummarizeService()

    print(rag.rag_summarize("擅自占用、挖掘公路的判定标准是什么"))
