import unittest

from agent.intents import IntentClassifier, IntentType


class IntentClassifierTests(unittest.TestCase):
    def setUp(self):
        self.classifier = IntentClassifier()

    def test_case_statistics(self):
        decision = self.classifier.classify("2026年7月23日哪个支队案件最多？")
        self.assertEqual(decision.primary_intent, IntentType.CASE_QUERY)
        self.assertIn("2026年7月23日", decision.entities.dates)

    def test_report_is_composite_intent(self):
        decision = self.classifier.classify("统计2026年5月密云执法队案件并生成执法报告")
        self.assertEqual(decision.primary_intent, IntentType.REPORT_GENERATION)
        self.assertEqual(
            decision.secondary_intents,
            [IntentType.CASE_QUERY, IntentType.KNOWLEDGE_QUERY],
        )
        self.assertIn("密云执法队", decision.entities.units)

    def test_uploaded_image(self):
        decision = self.classifier.classify(
            "图中行为是否属于道路养护？【上传图片：图片ID img_ab12cd34】"
        )
        self.assertEqual(decision.primary_intent, IntentType.IMAGE_ASSESSMENT)
        self.assertEqual(decision.entities.image_ids, ["img_ab12cd34"])

    def test_image_case_search_does_not_become_assessment(self):
        decision = self.classifier.classify(
            "查找一下有没有类似的历史案例【上传图片：图片ID img_case001】"
        )
        self.assertEqual(decision.primary_intent, IntentType.IMAGE_CASE_SEARCH)

    def test_explicit_image_violation_question_stays_assessment(self):
        decision = self.classifier.classify(
            "结合相似案例判断这张图是否违法【上传图片：图片ID img_case002】"
        )
        self.assertEqual(decision.primary_intent, IntentType.IMAGE_ASSESSMENT)

    def test_find_violation_cases_is_case_search_not_current_image_judgement(self):
        decision = self.classifier.classify(
            "看这张图有没有违法案例【上传图片：图片ID img_case003】"
        )
        self.assertEqual(decision.primary_intent, IntentType.IMAGE_CASE_SEARCH)

    def test_current_image_has_violation_is_assessment(self):
        decision = self.classifier.classify(
            "看一下图中有没有违法【上传图片：图片ID img_case004】"
        )
        self.assertEqual(decision.primary_intent, IntentType.IMAGE_ASSESSMENT)

    def test_compound_image_request_creates_dependent_goals(self):
        decision = self.classifier.classify(
            "看下这张图片有没有历史案例，并根据历史案例的描述以及判决分析下这个图片"
            "有没有擅自占用公路违法行为【上传图片：图片ID img_multi01】"
        )
        self.assertEqual(decision.primary_intent, IntentType.IMAGE_ASSESSMENT)
        self.assertEqual(
            [goal.intent for goal in decision.goals],
            [IntentType.IMAGE_CASE_SEARCH, IntentType.IMAGE_ASSESSMENT],
        )
        self.assertEqual(
            decision.goals[1].depends_on,
            [IntentType.IMAGE_CASE_SEARCH],
        )

    def test_knowledge_query(self):
        decision = self.classifier.classify("道路养护行为的判定标准是什么？")
        self.assertEqual(decision.primary_intent, IntentType.KNOWLEDGE_QUERY)

    def test_statistics_and_knowledge_are_two_independent_goals(self):
        decision = self.classifier.classify("统计7月案件数量，并说明相关法规依据")
        self.assertEqual(
            [goal.intent for goal in decision.goals],
            [IntentType.CASE_QUERY, IntentType.KNOWLEDGE_QUERY],
        )


if __name__ == "__main__":
    unittest.main()
