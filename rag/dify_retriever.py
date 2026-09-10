"""
Dify 知识库检索客户端：调用 Dify 的 Dataset 检索 API（POST /v1/datasets/{dataset_id}/retrieve），
把返回的片段转成与本地 Chroma 兼容的 langchain Document 列表，供 rag_summarize 使用。

未配置 DIFY_API_KEY 或 dataset_id 时 available=False，调用方应回退本地向量库。
"""
import os
import time
from dotenv import load_dotenv
load_dotenv()  #  单独运行本模块时也能读到 .env 中的 DIFY_API_KEY
import requests
from langchain_core.documents import Document
from utils.config_handler import dify_conf
from utils.logger_handler import logger


def normalize_retrieval_query(query: str, max_chars: int = 240) -> str:
    """压缩空白并限制 Dify 查询长度，避免接口返回 string_too_long。"""
    normalized = " ".join(str(query).split()).strip()
    if max_chars <= 0:
        raise ValueError("max_query_chars 必须大于 0")
    return normalized[:max_chars]


class DifyRetriever:
    def __init__(self):
        self.api_base = str(dify_conf.get("api_base", "https://api.dify.ai/v1")).rstrip("/")
        self.dataset_id = str(dify_conf.get("dataset_id", "")).strip()
        self.api_key = os.getenv("DIFY_API_KEY", "").strip()
        self.search_method = str(dify_conf.get("search_method", "semantic_search"))
        self.top_k = int(dify_conf.get("top_k", 5))
        self.score_threshold = float(dify_conf.get("score_threshold", 0.0))
        self.score_threshold_enabled = bool(dify_conf.get("score_threshold_enabled", False))
        self.max_query_chars = min(int(dify_conf.get("max_query_chars", 240)), 250)
        self.connect_timeout = max(float(dify_conf.get("connect_timeout_seconds", 5)), 1)
        self.read_timeout = max(float(dify_conf.get("read_timeout_seconds", 60)), 5)
        self.max_retries = min(max(int(dify_conf.get("max_retries", 1)), 0), 2)
        self.retry_backoff = max(float(dify_conf.get("retry_backoff_seconds", 1)), 0)
        self.last_error: str | None = None
        self.last_error_code: str | None = None
        self.last_attempts: int = 0

    @property
    def available(self) -> bool:
        return bool(self.api_key and self.dataset_id and self.api_base)

    def retrieve(self, query: str, doc_type: str = "") -> list[Document]:
        """
        检索知识库。
        :param query: 检索词
        :param doc_type: 按元数据 doc_type 过滤，可选 "standard"（判定标准/法规）或 "case"（案例）；空则不过滤
        """
        self.last_error = None
        self.last_error_code = None
        self.last_attempts = 0
        if not self.available:
            self.last_error_code = "DIFY_NOT_CONFIGURED"
            self.last_error = "Dify API 未配置完整"
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

        retrieval_query = normalize_retrieval_query(query, self.max_query_chars)
        if retrieval_query != " ".join(str(query).split()).strip():
            logger.warning(
                "[DifyRetriever]检索词超过 Dify 限制，已从 %d 字符压缩到 %d 字符",
                len(" ".join(str(query).split()).strip()),
                len(retrieval_query),
            )

        payload = {
            "query": retrieval_query,
            "retrieval_model": retrieval_model,
        }

        data = None
        for attempt in range(1, self.max_retries + 2):
            self.last_attempts = attempt
            try:
                resp = requests.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=(self.connect_timeout, self.read_timeout),
                )
                if resp.status_code in {502, 503, 504} and attempt <= self.max_retries:
                    logger.warning(
                        "[DifyRetriever]第 %d 次检索返回 HTTP %d，%.1f 秒后重试",
                        attempt,
                        resp.status_code,
                        self.retry_backoff,
                    )
                    if self.retry_backoff:
                        time.sleep(self.retry_backoff)
                    continue
                resp.raise_for_status()
                data = resp.json()
                self.last_error = None
                self.last_error_code = None
                break
            except requests.Timeout:
                self.last_error_code = "DIFY_TIMEOUT"
                self.last_error = (
                    f"Dify 检索读取超时（连接 {self.connect_timeout:g}s / "
                    f"读取 {self.read_timeout:g}s，第 {attempt} 次）"
                )
                if attempt <= self.max_retries:
                    logger.warning("[DifyRetriever]%s，%.1f 秒后重试", self.last_error, self.retry_backoff)
                    if self.retry_backoff:
                        time.sleep(self.retry_backoff)
                    continue
                logger.error("[DifyRetriever]%s，已停止重试", self.last_error)
                return []
            except requests.ConnectionError as exc:
                self.last_error_code = "DIFY_CONNECTION_ERROR"
                self.last_error = f"Dify 连接失败：{exc}"
                if attempt <= self.max_retries:
                    logger.warning("[DifyRetriever]%s，%.1f 秒后重试", self.last_error, self.retry_backoff)
                    if self.retry_backoff:
                        time.sleep(self.retry_backoff)
                    continue
                logger.error("[DifyRetriever]%s，已停止重试", self.last_error)
                return []
            except requests.HTTPError as exc:
                response = exc.response
                body = response.text[:2000] if response is not None else ""
                status = response.status_code if response is not None else "unknown"
                self.last_error_code = f"DIFY_HTTP_{status}"
                self.last_error = f"Dify 检索返回 HTTP {status}：{body}"
                logger.error("[DifyRetriever]%s", self.last_error)
                return []
            except Exception as exc:
                self.last_error_code = "DIFY_ERROR"
                self.last_error = f"Dify 检索异常：{exc}"
                logger.error("[DifyRetriever]%s", self.last_error, exc_info=True)
                return []

        if data is None:
            self.last_error_code = "DIFY_NO_RESPONSE"
            self.last_error = "Dify 检索未取得有效响应"
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

        logger.info(f"[DifyRetriever]查询「{retrieval_query}」(doc_type={doc_type or '全部'})返回 {len(documents)} 条")
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
