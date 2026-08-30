"""추론이 max_tokens를 다 먹어 본문이 빈 응답은 executor가 예산을 늘려 다시 부른다."""
import unittest
from unittest import mock

from llm import call_registry as cr


def _profile(**extra):
    return cr.CallSiteProfile(
        feature="t", provider="deepseek_anthropic", model="deepseek-v4-pro",
        temperature=0.2, max_tokens=20000, timeout=60.0, json_mode=False,
        note="", managed="executor", model_env_override=None, extra=extra)


class OutputBudgetTests(unittest.TestCase):
    def test_complete_reply_passes_through(self):
        calls = []
        def call(mt):
            calls.append(mt); return "본문", {"tokens_out": 100}, False
        text, usage = cr._with_output_budget(_profile(thinking={"type": "enabled"}), call)
        self.assertEqual(text, "본문"); self.assertEqual(calls, [20000])

    def test_empty_reply_at_limit_escalates_and_recovers(self):
        calls = []
        def call(mt):
            calls.append(mt)
            if mt < 40000:
                return "", {"tokens_out": mt}, True
            return "완결된 본문", {"tokens_out": 22200}, False
        with self.assertLogs("llm.call_registry", level="WARNING") as logs:
            text, _ = cr._with_output_budget(_profile(thinking={"type": "enabled"}), call)
        self.assertEqual(text, "완결된 본문")
        self.assertEqual(calls, [20000, 40000])
        self.assertIn("output budget exhausted", logs.output[0])

    def test_truncated_reasoning_reply_escalates(self):
        # 추론이 켜진 호출은 본문이 있어도 길이에 걸렸으면 잘린 답이다.
        calls = []
        def call(mt):
            calls.append(mt)
            return ("잘린", {"tokens_out": mt}, True) if mt < 40000 else ("온전한", {}, False)
        text, _ = cr._with_output_budget(_profile(thinking={"type": "enabled"}), call)
        self.assertEqual(text, "온전한"); self.assertEqual(calls, [20000, 40000])

    def test_no_reasoning_truncated_text_is_kept(self):
        # 추론이 꺼진 호출의 길이 상한은 호출부의 선택일 수 있다 — 경고만.
        calls = []
        def call(mt):
            calls.append(mt); return "짧게 잘린 요약", {"tokens_out": mt}, True
        with self.assertLogs("llm.call_registry", level="WARNING"):
            text, _ = cr._with_output_budget(_profile(thinking={"type": "disabled"}), call)
        self.assertEqual(text, "짧게 잘린 요약"); self.assertEqual(calls, [20000])

    def test_raises_after_cap(self):
        calls = []
        def call(mt):
            calls.append(mt); return "", {"tokens_out": mt}, True
        with self.assertLogs("llm.call_registry", level="WARNING"):
            with self.assertRaises(cr.OutputBudgetExhausted) as ctx:
                cr._with_output_budget(_profile(thinking={"type": "enabled"}), call)
        self.assertEqual(calls, [20000, 40000, 65536])
        self.assertIn("출력 예산 소진", str(ctx.exception))

    def test_generate_sync_reports_exhaustion_as_failure(self):
        # 조용한 None이 아니라 실패로 기록되고 원인이 로그에 남는다.
        p = _profile(thinking={"type": "enabled"})
        def boom(profile, prompt, system):
            raise cr.OutputBudgetExhausted("x: 출력 예산 소진")
        with mock.patch.object(cr, "resolve", return_value=p), \
             mock.patch.dict(cr._EXECUTORS, {"deepseek_anthropic": boom}), \
             mock.patch("llm.gateway.check_llm_call"), \
             mock.patch("llm.gateway.record_llm_call") as rec:
            with self.assertLogs("llm.call_registry", level="WARNING") as logs:
                self.assertIsNone(cr.generate_sync("t", "p"))
        self.assertEqual(rec.call_args.kwargs["status"], "error")
        self.assertIn("출력 예산 소진", logs.output[-1])


if __name__ == "__main__":
    unittest.main()
