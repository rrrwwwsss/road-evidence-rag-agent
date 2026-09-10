"""不调用大模型的 Intent 规则层评测。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.intents import IntentClassifier


def main() -> int:
    test_path = Path(__file__).with_name("intent_test_set.jsonl")
    classifier = IntentClassifier()
    total = 0
    passed = 0
    for line in test_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        actual = classifier.classify(case["question"]).primary_intent.value
        total += 1
        if actual == case["expected_intent"]:
            passed += 1
        else:
            print(f"FAIL {case['id']}: expected={case['expected_intent']} actual={actual}")
    accuracy = passed / total if total else 0
    print(f"Intent accuracy: {passed}/{total} = {accuracy:.2%}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
