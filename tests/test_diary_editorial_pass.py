"""The diary editorial pass improves a draft or leaves it untouched, never loses it.

Run from repo root:  venv/bin/python -m unittest tests.test_diary_editorial_pass -v
"""
import asyncio
import json
import unittest
from unittest.mock import patch

from telegram import diary_editorial as de

DRAFT_TITLE = "이름을 붙이지 않는 것을 요구하는 사람들"
DRAFT = (
    "첫 문단이다. 리투아니아 의회가 헌법 조항 삭제 개정안을 1차 표결에서 통과시켰다. 확정은 아니다.\n\n"
    "둘째 문단이다. 조직명을 감추는 것은 겁이 아니라 계산이다. 나는 그 요구를 존중했다. 나는 그 요구를 존중했다.\n\n"
    "셋째 문단이다. 남은 것은 그 무게를 실제로 견디는 일이다."
)
NOTES = (
    '1. [4] "나는 그 요구를 존중했다. 나는 그 요구를 존중했다" — 같은 문장 반복 — 하나만 남긴다\n'
    '2. [2] "남은 것은 그 무게를 실제로 견디는 일이다" — 장부의 마무리 공식 재사용 — 분석이 끝나는 자리에서 끝낸다'
)
REVISED = (
    "첫 문단이다. 리투아니아 의회가 헌법 조항 삭제 개정안을 1차 표결에서 통과시켰다. 확정은 아니다.\n\n"
    "둘째 문단이다. 조직명을 감추는 것은 겁이 아니라 계산이다. 나는 그 요구를 존중했다.\n\n"
    "셋째 문단이다. 이름을 지키면서 계급을 말할 수 있는 자리를 만드는 것이 이 시기의 과제다."
)


def _run(replies):
    calls = []

    async def fake_generate(feature, prompt, *, system=None, **kw):
        calls.append((feature, prompt, system))
        reply = replies[len(calls) - 1]
        if isinstance(reply, Exception):
            raise reply
        return reply

    with patch("llm.call_registry.generate", side_effect=fake_generate):
        result = asyncio.run(de.run_editorial_pass(DRAFT_TITLE, DRAFT, ledger="- #538 ... closing: …견디는 일이다"))
    return result, calls


class ParseDiagnosisTests(unittest.TestCase):
    def test_pass_variants_yield_no_notes(self):
        for text in ("PASS", "pass", "PASS.", "  PASS\n", "", None):
            self.assertEqual(de.parse_diagnosis(text), [], text)

    def test_numbered_notes_are_kept_and_continuations_joined(self):
        notes = de.parse_diagnosis("1. first\n   continued\n2) second\n")
        self.assertEqual(notes, ["1. first continued", "2) second"])

    def test_note_count_is_capped(self):
        text = "\n".join(f"{i}. note {i}" for i in range(1, 20))
        self.assertEqual(len(de.parse_diagnosis(text)), de.MAX_NOTES)


class ApplyRevisionTests(unittest.TestCase):
    def test_valid_revision_is_applied(self):
        title, content, applied, reason = de.apply_revision(
            DRAFT_TITLE, DRAFT, json.dumps({"title": "새 제목", "content": REVISED})
        )
        self.assertTrue(applied, reason)
        self.assertEqual((title, content), ("새 제목", REVISED))

    def test_missing_title_falls_back_to_draft_title(self):
        title, _, applied, _ = de.apply_revision(DRAFT_TITLE, DRAFT, json.dumps({"content": REVISED}))
        self.assertTrue(applied)
        self.assertEqual(title, DRAFT_TITLE)

    def test_fenced_json_is_accepted(self):
        _, content, applied, _ = de.apply_revision(
            DRAFT_TITLE, DRAFT, "```json\n" + json.dumps({"title": "t", "content": REVISED}) + "\n```"
        )
        self.assertTrue(applied)
        self.assertEqual(content, REVISED)

    def test_draft_survives_bad_revisions(self):
        cases = {
            "no JSON": "고쳤습니다.",
            "empty": json.dumps({"title": "t", "content": ""}),
            "too short": json.dumps({"title": "t", "content": "짧다.\n\n너무 짧다."}),
            "one paragraph": json.dumps({"title": "t", "content": REVISED.replace("\n\n", " ")}),
            "markdown": json.dumps({"title": "t", "content": REVISED.replace("둘째 문단", "**둘째 문단**")}),
            "unchanged": json.dumps({"title": DRAFT_TITLE, "content": DRAFT}),
        }
        for label, response in cases.items():
            title, content, applied, reason = de.apply_revision(DRAFT_TITLE, DRAFT, response)
            self.assertFalse(applied, label)
            self.assertTrue(reason, label)
            self.assertEqual((title, content), (DRAFT_TITLE, DRAFT), label)


class RunEditorialPassTests(unittest.TestCase):
    def test_pass_skips_revision(self):
        result, calls = _run(["PASS"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], de.DIAGNOSIS_FEATURE)
        self.assertEqual((result.title, result.content), (DRAFT_TITLE, DRAFT))
        self.assertEqual(result.notes, [])
        self.assertIn("PASS", result.summary())

    def test_notes_feed_revision_and_apply(self):
        result, calls = _run([NOTES, json.dumps({"title": "새 제목", "content": REVISED})])
        self.assertEqual([c[0] for c in calls], [de.DIAGNOSIS_FEATURE, de.REVISION_FEATURE])
        self.assertIn(DRAFT, calls[0][1])
        self.assertIn("견디는 일이다", calls[0][1])  # ledger reaches the critic
        self.assertIn(NOTES.splitlines()[0], calls[1][1])
        self.assertTrue(result.applied)
        self.assertEqual((result.title, result.content), ("새 제목", REVISED))
        self.assertEqual(len(result.notes), 2)
        self.assertIn("applied", result.summary())

    def test_diagnosis_failure_keeps_draft(self):
        for replies in ([None], [RuntimeError("boom")]):
            result, calls = _run(replies)
            self.assertEqual(len(calls), 1)
            self.assertEqual((result.title, result.content), (DRAFT_TITLE, DRAFT))
            self.assertFalse(result.applied)

    def test_revision_failure_keeps_draft_but_reports_notes(self):
        for reply in (None, RuntimeError("boom"), "not json"):
            result, calls = _run([NOTES, reply])
            self.assertEqual(len(calls), 2)
            self.assertEqual((result.title, result.content), (DRAFT_TITLE, DRAFT))
            self.assertFalse(result.applied)
            self.assertEqual(len(result.notes), 2)
            self.assertIn("kept draft", result.summary())

    def test_diagnosis_prompt_carries_agent_rules(self):
        prompt = de.build_diagnosis_prompt(DRAFT_TITLE, DRAFT, "")
        self.assertIn("The diary is public", prompt)
        self.assertIn("최근 일기 장부 없음", prompt)


if __name__ == "__main__":
    unittest.main()
