import unittest
from unittest.mock import MagicMock, patch

import requests

from agent.skills.image_case_search.handler import build_case_retrieval_query
from rag.dify_retriever import DifyRetriever, normalize_retrieval_query


class RetrievalQueryLimitTests(unittest.TestCase):
    def test_dify_query_never_exceeds_configured_limit(self):
        query = "道路施工现场、施工车辆、交通锥、机械设备。" * 50
        normalized = normalize_retrieval_query(query, max_chars=240)
        self.assertLessEqual(len(normalized), 240)

    def test_whitespace_is_compacted_before_length_limit(self):
        normalized = normalize_retrieval_query("道路施工\n\n  交通锥   施工车辆", max_chars=240)
        self.assertEqual(normalized, "道路施工 交通锥 施工车辆")

    def test_image_case_query_prefers_explicit_visual_keywords(self):
        visual = (
            "这里是一段很长的图片描述。" * 30
            + "\n检索关键词：道路施工、黄色挖掘机、交通锥、钢结构支架"
        )
        query = build_case_retrieval_query(
            "查找类似案例【上传图片：图片ID img_abc123】",
            visual,
        )
        self.assertLessEqual(len(query), 220)
        self.assertIn("黄色挖掘机", query)
        self.assertNotIn("img_abc123", query)
        self.assertNotIn("这里是一段很长", query)

    @patch("rag.dify_retriever.time.sleep")
    @patch("rag.dify_retriever.requests.post")
    def test_timeout_retries_once_with_connect_and_read_tuple(self, post, sleep):
        response = MagicMock(status_code=200)
        response.json.return_value = {"records": []}
        post.side_effect = [requests.ReadTimeout("slow"), response]
        retriever = DifyRetriever()
        retriever.api_key = "test-key"
        retriever.dataset_id = "dataset-id"
        retriever.max_retries = 1
        retriever.retry_backoff = 0
        retriever.connect_timeout = 5
        retriever.read_timeout = 60
        self.assertEqual(retriever.retrieve("测试"), [])
        self.assertEqual(post.call_count, 2)
        self.assertEqual(post.call_args.kwargs["timeout"], (5, 60))
        self.assertIsNone(retriever.last_error)
        self.assertEqual(retriever.last_attempts, 2)

    @patch("rag.dify_retriever.time.sleep")
    @patch("rag.dify_retriever.requests.post")
    def test_final_timeout_records_machine_readable_reason(self, post, sleep):
        post.side_effect = requests.ReadTimeout("slow")
        retriever = DifyRetriever()
        retriever.api_key = "test-key"
        retriever.dataset_id = "dataset-id"
        retriever.max_retries = 1
        retriever.retry_backoff = 0
        retriever.retrieve("测试")
        self.assertEqual(post.call_count, 2)
        self.assertEqual(retriever.last_error_code, "DIFY_TIMEOUT")
        self.assertIn("读取超时", retriever.last_error)


if __name__ == "__main__":
    unittest.main()
