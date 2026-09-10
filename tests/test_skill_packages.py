import unittest
from pathlib import Path

from agent.intents import IntentType
from agent.skills.package import load_skill_package
from agent.skills.registry import SkillRegistry


class SkillPackageTests(unittest.TestCase):
    PACKAGE_NAMES = (
        "case_query",
        "knowledge_query",
        "image_case_search",
        "image_assessment",
        "report_generation",
    )

    def test_all_skill_packages_have_instructions_prompt_and_handler(self):
        skills_root = Path(__file__).resolve().parents[1] / "agent" / "skills"
        for name in self.PACKAGE_NAMES:
            root = skills_root / name
            package = load_skill_package(root)
            self.assertTrue(package.instructions)
            self.assertTrue(package.prompt)
            self.assertTrue((root / "handler.py").is_file())

    def test_registry_maps_every_business_intent_to_packaged_skill(self):
        registry = SkillRegistry.default(model=None)
        expected = {
            IntentType.CASE_QUERY,
            IntentType.KNOWLEDGE_QUERY,
            IntentType.IMAGE_CASE_SEARCH,
            IntentType.IMAGE_ASSESSMENT,
            IntentType.REPORT_GENERATION,
        }
        for intent in expected:
            skill = registry.get(intent)
            self.assertIsNotNone(skill)
            self.assertEqual(skill.package.name, intent.value)


if __name__ == "__main__":
    unittest.main()
