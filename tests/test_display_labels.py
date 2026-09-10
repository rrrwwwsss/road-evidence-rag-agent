import unittest

from agent.display_labels import localize_trace


class DisplayLabelTests(unittest.TestCase):
    def test_localizes_saved_english_trace(self):
        self.assertEqual(
            localize_trace("意图识别完成：image_assessment，置信度 96%"),
            "意图识别完成：图片违法与养护判定，置信度 96%",
        )

    def test_localizes_image_case_search(self):
        self.assertEqual(
            localize_trace("已选择业务能力：image_case_search"),
            "已选择业务能力：图片相似案例检索",
        )


if __name__ == "__main__":
    unittest.main()
