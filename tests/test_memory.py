import unittest

from agent.memory import (
    MAX_MESSAGE_CHARS,
    MAX_RECENT_MESSAGES,
    append_recent_turn,
    trim_recent_messages,
)


class RecentMessageMemoryTests(unittest.TestCase):
    def test_keeps_only_eight_user_assistant_rounds(self):
        messages = []
        for index in range(10):
            messages.extend([
                {"role": "user", "content": f"问题 {index}"},
                {"role": "assistant", "content": f"回答 {index}"},
            ])
        recent = trim_recent_messages(messages)
        self.assertEqual(len(recent), MAX_RECENT_MESSAGES)
        self.assertEqual(recent[0]["content"], "问题 2")
        self.assertEqual(recent[-1]["content"], "回答 9")

    def test_ignores_invalid_roles_and_empty_messages(self):
        recent = trim_recent_messages([
            {"role": "system", "content": "系统提示词"},
            {"role": "user", "content": ""},
            {"role": "user", "content": "有效问题"},
        ])
        self.assertEqual(recent, [{"role": "user", "content": "有效问题"}])

    def test_long_message_is_bounded_but_preserves_its_end(self):
        content = "前" * MAX_MESSAGE_CHARS + "图片ID img_keep_me"
        recent = trim_recent_messages([{"role": "user", "content": content}])
        self.assertLessEqual(len(recent[0]["content"]), MAX_MESSAGE_CHARS + 20)
        self.assertTrue(recent[0]["content"].endswith("图片ID img_keep_me"))

    def test_append_turn_keeps_both_sides(self):
        recent = append_recent_turn([], "本轮问题", "本轮回答")
        self.assertEqual(recent, [
            {"role": "user", "content": "本轮问题"},
            {"role": "assistant", "content": "本轮回答"},
        ])


if __name__ == "__main__":
    unittest.main()
