"""Structured result envelope for delegated agent tasks."""

import asyncio
import json
import unittest
from unittest import mock

from telegram import result_envelope as env


class NormalizeTest(unittest.TestCase):
    def test_valid_done_envelope_is_bounded(self):
        out = env.normalize(
            "DONE", "  answer  ",
            findings=[{"claim": "c1", "sources": ["https://a", "", "b"]}, "plain claim", {"x": 1}],
            open_questions=["q"], artifacts=["slug:x"],
        )
        self.assertEqual(out["status"], "done")
        self.assertEqual(out["answer"], "answer")
        self.assertEqual(out["findings"], [
            {"claim": "c1", "sources": ["https://a", "b"]},
            {"claim": "plain claim", "sources": []},
        ])
        self.assertEqual(out["blockers"], [])

    def test_contract_breaches_raise(self):
        with self.assertRaises(ValueError):
            env.normalize("finished", "a")
        with self.assertRaises(ValueError):
            env.normalize("done", "   ")
        with self.assertRaises(ValueError):
            env.normalize("blocked", "a")
        with self.assertRaises(ValueError):
            env.normalize("partial", "a")
        self.assertEqual(env.normalize("blocked", "a", blockers=["CAPTCHA"])["status"], "blocked")

    def test_limits(self):
        out = env.normalize("done", "x" * 9000,
                            findings=[{"claim": str(i)} for i in range(50)],
                            open_questions=["q" * 900] * 20)
        self.assertEqual(len(out["answer"]), env.MAX_ANSWER + 1)
        self.assertEqual(len(out["findings"]), env.MAX_FINDINGS)
        self.assertEqual(len(out["open_questions"]), env.MAX_LIST)
        self.assertEqual(len(out["open_questions"][0]), env.MAX_ITEM + 1)


class LoadRenderTest(unittest.TestCase):
    ENVELOPE = {"status": "partial", "answer": "half", "findings": [{"claim": "c", "sources": []}],
                "open_questions": ["rest"], "blockers": [], "artifacts": []}

    def test_load_from_row_json_and_dict(self):
        self.assertEqual(env.load({"metadata": {"result_envelope": self.ENVELOPE}}), self.ENVELOPE)
        self.assertEqual(env.load({"metadata": json.dumps({"result_envelope": self.ENVELOPE})}),
                         self.ENVELOPE)
        self.assertIsNone(env.load({"metadata": None}))
        self.assertIsNone(env.load({"metadata": {"result_envelope": {"status": "weird"}}}))
        self.assertIsNone(env.load({"metadata": "not json"}))

    def test_render_flags_mismatches(self):
        text = env.render(self.ENVELOPE, execution_status="done")
        self.assertIn("agent-reported status: partial", text)
        self.assertIn("treat the goal as not fully met", text)
        self.assertIn("no source given", text)
        self.assertIn("open questions:\n  - rest", text)
        done = dict(self.ENVELOPE, status="done")
        self.assertIn("independent verification failed",
                      env.render(done, execution_status="done", verification_status="failed"))
        self.assertEqual(env.render(None), "")

    def test_attach_rendered_feeds_prompt_previews(self):
        from llm.prompt_context import format_dependency_results
        rows = [
            {"id": 1, "agent_type": "analyst", "content": "t", "result": "free text",
             "status": "done", "metadata": {"result_envelope": self.ENVELOPE}},
            {"id": 2, "agent_type": "scout", "content": "t", "result": "plain", "status": "done",
             "metadata": None},
        ]
        block = format_dependency_results(env.attach_rendered(rows), "claude")
        self.assertIn("agent-reported status: partial", block)
        self.assertIn("[Free-text report]\nfree text", block)
        self.assertNotIn("result_envelope_text", block)
        self.assertNotIn("result_envelope_text", rows[1])


class HandlerTest(unittest.TestCase):
    def test_handler_saves_and_rejects(self):
        from tool_gateway.results import ToolRejection
        handler = env.make_handler(42)
        with mock.patch.object(env, "save") as save:
            out = asyncio.run(handler("done", "ok", findings=[{"claim": "c", "sources": ["u"]}]))
        save.assert_called_once()
        self.assertEqual(save.call_args[0][0], 42)
        self.assertIn("status=done", out)
        with self.assertRaises(ToolRejection):
            asyncio.run(handler("blocked", "no"))

    def test_save_failure_is_tool_failure(self):
        from tool_gateway.results import is_failure
        handler = env.make_handler(1)
        with mock.patch.object(env, "save", side_effect=RuntimeError("db down")):
            out = asyncio.run(handler("done", "ok"))
        self.assertTrue(is_failure(out))

    def test_registered_in_task_context_tools_and_policy(self):
        from security_gateway.policy import risk_class
        from self_runtime.tools import build_task_context_tools
        tools, handlers = build_task_context_tools(7, 1)
        self.assertIn("report_result", {t["name"] for t in tools})
        self.assertIn("report_result", handlers)
        self.assertEqual(risk_class("report_result"), "state")

    def test_schema_accepts_handler_arguments(self):
        from tool_gateway.validation import validate_tool_arguments
        args = {"status": "partial", "answer": "a", "findings": [{"claim": "c", "sources": ["s"]}],
                "open_questions": ["q"], "blockers": [], "artifacts": []}
        validate_tool_arguments("report_result", args, schema=env.TOOL["input_schema"],
                                risk_class="state")


if __name__ == "__main__":
    unittest.main()
