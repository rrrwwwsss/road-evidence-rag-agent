import unittest

from utils.message_content import extract_text


class MessageContentTests(unittest.TestCase):
    def test_plain_string(self):
        self.assertEqual(extract_text("识别结果"), "识别结果")

    def test_qwen_text_block_list(self):
        content = [{"text": "### 识别结论\n属于道路养护施工场景。"}]
        self.assertEqual(
            extract_text(content),
            "### 识别结论\n属于道路养护施工场景。",
        )

    def test_multiple_content_blocks(self):
        content = [{"text": "第一段"}, {"output_text": "第二段"}]
        self.assertEqual(extract_text(content), "第一段\n第二段")


if __name__ == "__main__":
    unittest.main()
