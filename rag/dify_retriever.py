"""
Dify 知识库检索客户端：调用 Dify 的 Dataset 检索 API（POST /v1/datasets/{dataset_id}/retrieve），
把返回的片段转成与本地 Chroma 兼容的 langchain Document 列表，供 rag_summarize 使用。

未配置 DIFY_API_KEY 或 dataset_id 时 available=False，调用方应回退本地向量库。
"""
import os
from dotenv import load_dotenv
load_dotenv()  #  单独运行本模块时也能读到 .env 中的 DIFY_API_KEY
import requests
from langchain_core.documents import Document
from utils.config_handler import dify_conf
from utils.logger_handler import logger


class DifyRetriever:
    def __init__(self):
        self.api_base = str(dify_conf.get("api_base", "https://api.dify.ai/v1")).rstrip("/")
        self.dataset_id = str(dify_conf.get("dataset_id", "")).strip()
        self.api_key = os.getenv("DIFY_API_KEY", "").strip()
        self.search_method = str(dify_conf.get("search_method", "semantic_search"))
        self.top_k = int(dify_conf.get("top_k", 5))
        self.score_threshold = float(dify_conf.get("score_threshold", 0.0))
        self.score_threshold_enabled = bool(dify_conf.get("score_threshold_enabled", False))

    @property
    def available(self) -> bool:
        return bool(self.api_key and self.dataset_id and self.api_base)

    def retrieve(self, query: str, doc_type: str = "") -> list[Document]:
        """
        检索知识库。
        :param query: 检索词
        :param doc_type: 按元数据 doc_type 过滤，可选 "standard"（判定标准/法规）或 "case"（案例）；空则不过滤
        """
        if not self.available:
            logger.warning("[DifyRetriever]未配置 DIFY_API_KEY / dataset_id / api_base，跳过 Dify 检索")
            return []

        url = f"{self.api_base}/datasets/{self.dataset_id}/retrieve"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        retrieval_model = {
            "search_method": self.search_method,
            "reranking_enable": False,
            "top_k": self.top_k,
            "score_threshold_enabled": self.score_threshold_enabled,
            "score_threshold": self.score_threshold,
        }

        if doc_type in ("standard", "case"):
            retrieval_model["metadata_filter"] = {
                "metadata_field": "doc_type",
                "metadata_type": "string",
                "metadata_value": doc_type,
            }

        payload = {
            "query": query,
            "retrieval_model": retrieval_model,
        }

        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error(f"[DifyRetriever]检索失败：{e}", exc_info=True)
            return []

        documents = []
        for record in data.get("records", []):
            segment = record.get("segment") or {}
            content = segment.get("content", "")
            if not content:
                continue

            metadata = {
                "source": (segment.get("document") or {}).get("name", ""),
                "score": record.get("score"),
            }
            if segment.get("document_id"):
                metadata["document_id"] = segment["document_id"]

            documents.append(Document(page_content=content, metadata=metadata))

        logger.info(f"[DifyRetriever]查询「{query}」(doc_type={doc_type or '全部'})返回 {len(documents)} 条")
        return documents


if __name__ == '__main__':
    retriever = DifyRetriever()
    print("DifyRetriever available:", retriever.available)
    if retriever.available:
        docs = retriever.retrieve("擅自占用、挖掘公路")
        for doc in docs[:3]:
            print("-" * 20)
            print(doc.metadata)
            print(doc.page_content[:120])