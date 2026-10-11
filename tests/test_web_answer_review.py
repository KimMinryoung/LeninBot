"""Deliverable review, bounded repair and privacy regressions without external I/O."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from llm.call_registry import GenerationResult
from services import web_answer_review as review


class ReviewTests(unittest.IsolatedAsyncioTestCase):
    async def check(self, output, *, answer="완전한 답변입니다.", metadata=None, evidence=()):
        with patch.object(review.call_registry, "resolve", return_value=SimpleNamespace(timeout=1)), \
                patch.object(review.call_registry, "generate_detailed", return_value=output) as generate:
            result = await review.review_web_answer(
                message="한잎이 두잎을 설득했다. 이 전개를 분석해줘.", answer=answer,
                history=[{"role": "user", "content": "original"}], evidence=list(evidence),
                metadata=metadata or {"complete": True},
            )
        return result, generate

    async def test_valid_short_answer_is_not_rejected_by_length_or_punctuation(self):
        for answer in ("예", "아니요", "42", "레닌"):
            issues, _ = await self.check(GenerationResult(text='{"issues": []}'), answer=answer)
            self.assertEqual(issues, [])

    async def test_incomplete_provider_output_never_passes(self):
        issues, _ = await self.check(GenerationResult(text='{"issues": []}'),
                                     metadata={"complete": False, "truncated": True})
        self.assertEqual(issues[0]["kind"], "incomplete")

    async def test_empty_answer_is_not_reviewed_or_approved(self):
        issues, gen = await self.check(GenerationResult(text='{"issues": []}'), answer="")
        self.assertEqual(issues[0]["kind"], "incomplete")
        gen.assert_not_called()

    async def test_malformed_unavailable_and_truncated_review_never_approve(self):
        for result in (
            GenerationResult(error_kind="transport"),
            GenerationResult(text='{"issues": []}', truncated=True),
            GenerationResult(text="PASS"),
            GenerationResult(text='{"issues": false}'),
            GenerationResult(text='{"issues": [], "unknown": true}'),
            GenerationResult(text='{"issues": [{"kind": [], "reason": "invalid"}]}'),
            GenerationResult(text='{"issues": [{"kind": "opinion", "reason": "different politics"}]}'),
        ):
            with self.subTest(result=result), self.assertRaises(review.WebAnswerReviewError):
                await self.check(result)

    async def test_exact_input_and_excerpt_coverage_reach_the_reviewer(self):
        _, gen = await self.check(GenerationResult(text='{"issues": []}'),
                                 evidence=["web_search: snippet" + "x" * 7000])
        payload = json.loads(gen.call_args.args[1])
        self.assertIn("한잎이 두잎을", payload["user_message"])
        self.assertTrue(payload["observed_tools"][0]["excerpted"])
        self.assertIn("web_search", payload["observed_tools"][0]["text"])
        self.assertIn("snippets, not full pages", gen.call_args.kwargs["system"])


class RepairTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from services import web_chat
        self.web = web_chat
        self.turn = web_chat._PreparedWebChat(
            original_message="비숑이 내 대화를 읽을 수 있음?", regeneration_source=None,
            history=[{"role": "user", "content": "비숑이 내 대화를 읽을 수 있음?"}],
            profile=SimpleNamespace(budget_usd=.3), system_prompt="persona",
            feedback_ids=[], preflight_tool_detail="",
        )
        self.issue = {"kind": "incomplete", "reason": "읽을 수 있 ends mid-sentence"}
        self.tracker = {"total_cost": .1, "rounds_used": 1,
                        "tool_work_details": ["web_search: original snippet"]}
        self.metadata = {"text": "읽을 수 있", "complete": True}
        self.progress = AsyncMock()

    async def run_repair(self):
        return await self.web._review_and_repair_web_answer(
            self.turn, self.metadata, tools=[], tool_handlers={},
            on_progress=self.progress, budget_tracker=self.tracker,
        )

    async def test_one_repair_uses_remaining_budget_and_keeps_both_attempts(self):
        async def repair(turn, **kwargs):
            self.assertAlmostEqual(kwargs["budget_usd"], .2)
            self.assertIn("original_request", turn.history[-1]["content"])
            kwargs["budget_tracker"].update(total_cost=.04, rounds_used=2,
                                            tool_work_details=["fetch_url: actual body"])
            return {"text": "관리자는 대화 기록을 읽을 수 있다.", "complete": True}
        with patch.object(self.web, "review_web_answer", new=AsyncMock(side_effect=[[self.issue], []])) as check, \
                patch.object(self.web, "_invoke_web_model", new=AsyncMock(side_effect=repair)) as invoke:
            result = await self.run_repair()
        invoke.assert_awaited_once()
        self.assertEqual(check.await_count, 2)
        self.assertEqual(self.tracker["answer_repairs"], 1)
        self.assertAlmostEqual(self.tracker["total_cost"], .14)
        self.assertEqual(len(self.tracker["tool_work_details"]), 2)
        self.assertEqual(self.tracker["final_response"], result["text"])
        self.assertEqual(len(self.turn.history), 1)

    async def test_bad_repair_stops_after_one_attempt(self):
        with patch.object(self.web, "review_web_answer", new=AsyncMock(return_value=[self.issue])), \
                patch.object(self.web, "_invoke_web_model", new=AsyncMock(return_value=self.metadata)) as invoke:
            with self.assertRaises(review.WebAnswerReviewError):
                await self.run_repair()
            invoke.assert_awaited_once()

    async def test_reviewer_outage_and_exhausted_budget_do_not_publish_or_retry(self):
        with patch.object(self.web, "review_web_answer", new=AsyncMock(side_effect=review.WebAnswerReviewError("offline"))), \
                patch.object(self.web, "_invoke_web_model", new=AsyncMock()) as invoke:
            with self.assertRaises(review.WebAnswerReviewError):
                await self.run_repair()
            invoke.assert_not_awaited()
        self.tracker["total_cost"] = .3
        with patch.object(self.web, "review_web_answer", new=AsyncMock(return_value=[self.issue])), \
                patch.object(self.web, "_invoke_web_model", new=AsyncMock()) as invoke:
            with self.assertRaises(review.WebAnswerReviewError):
                await self.run_repair()
            invoke.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
