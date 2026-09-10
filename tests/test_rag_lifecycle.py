import unittest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from rag.rag_service import RagSummarizeService
from rag.vector_store import get_vector_store_service


class RagLifecycleTests(unittest.TestCase):
    def tearDown(self):
        get_vector_store_service.cache_clear()

    @patch("rag.rag_service.get_vector_store_service")
    def test_chroma_is_not_initialized_during_rag_service_startup(self, vector_factory):
        service = RagSummarizeService()
        vector_factory.assert_not_called()
        self.assertIsNone(service._local_retriever)

    @patch("rag.rag_service.get_vector_store_service")
    def test_dify_success_does_not_initialize_chroma(self, vector_factory):
        service = RagSummarizeService()
        service.dify_retriever = MagicMock(available=True)
        service.dify_retriever.retrieve.return_value = [
            Document(page_content="Dify 命中结果", metadata={"source": "case.md"})
        ]
        service.dify_retriever.last_error = None
        docs = service.retriever_docs("测试")
        self.assertEqual(len(docs), 1)
        self.assertEqual(service.last_backend, "dify")
        vector_factory.assert_not_called()

    @patch("rag.rag_service.get_vector_store_service")
    def test_dify_timeout_records_reason_before_local_fallback(self, vector_factory):
        local_retriever = MagicMock()
        local_retriever.invoke.return_value = [Document(page_content="本地结果")]
        vector_factory.return_value.get_retriever.return_value = local_retriever
        service = RagSummarizeService()
        service.dify_retriever = MagicMock(available=True)
        service.dify_retriever.retrieve.return_value = []
        service.dify_retriever.last_error = "Dify 检索读取超时"
        docs = service.retriever_docs("测试")
        self.assertEqual(len(docs), 1)
        self.assertEqual(service.last_backend, "local_chroma")
        self.assertEqual(service.last_retrieval_error, "Dify 检索读取超时")

    @patch("rag.rag_service.get_vector_store_service")
    def test_chroma_initialization_failure_degrades_to_empty_results(self, vector_factory):
        service = RagSummarizeService()
        service.dify_retriever = MagicMock(available=False)
        vector_factory.side_effect = AttributeError(
            "'RustBindingsAPI' object has no attribute 'bindings'"
        )
        self.assertEqual(service.retriever_docs("测试"), [])

    @patch("rag.vector_store.VectorStoreService")
    def test_vector_store_factory_reuses_one_process_instance(self, service_class):
        get_vector_store_service.cache_clear()
        instance = object()
        service_class.return_value = instance
        self.assertIs(get_vector_store_service(), instance)
        self.assertIs(get_vector_store_service(), instance)
        service_class.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
