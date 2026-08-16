"""
总结服务类：用户提问，搜索参考资料，将提问和参考资料提交给模型，让模型总结回复
"""
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from rag.vector_store import VectorStoreService
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
        self.vector_store = VectorStoreService()
        self.retriever = self.vector_store.get_retriever()
        # Dify 知识库检索（未配置 DIFY_API_KEY / dataset_id 时自动回退本地向量库）
        self.dify_retriever = DifyRetriever()
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self._init_chain()

    def _init_chain(self):
        chain = self.prompt_template | print_prompt | self.model | StrOutputParser()
        return chain

    def retriever_docs(self, query: str, doc_type: str = "") -> list[Document]:
        if self.dify_retriever.available:
            docs = self.dify_retriever.retrieve(query, doc_type=doc_type)
            if docs:
                return docs
            logger.warning("[rag_service]Dify检索无结果，回退本地向量库")

        return self.retriever.invoke(query)

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