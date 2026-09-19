"""route_task: System One classifier first, LLM classifier when it is not confident."""
import json
import unittest
from unittest.mock import AsyncMock, patch

from llm import call_registry as cr
from self_runtime import tools


def decision(agent, confidence, second=("analyst", 0.05), routing_class="code_config_work", needs_id=0.2):
    return cr.Decision(answers={
        "agent": {"type": "choice", "choice": agent, "confidence": confidence,
                  "probabilities": {agent: confidence, second[0]: second[1]}},
        "routing_class": {"type": "choice", "choice": routing_class, "confidence": 0.9,
                          "probabilities": {routing_class: 0.9}},
        "needs_identifier": {"type": "noul", "noul": needs_id},
    }, model="typesafe/jev-1.13-test")


PROFILE = cr.CallSiteProfile(feature="task_routing_decision", provider="openrouter", model="typesafe/jev-1.13",
                             extra={"thresholds": {"accept": 0.85}})


class RouteTaskJevTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        p = patch.object(cr, "resolve", return_value=PROFILE)
        p.start(); self.addCleanup(p.stop)

    async def test_confident_decision_is_used_without_llm(self):
        with patch.object(cr, "decide", AsyncMock(return_value=decision("programmer", 0.97))) as decide, \
             patch.object(tools, "_classify_route_with_llm", AsyncMock()) as llm:
            rec, engine = await tools._classify_route("fix the scheduler", None)
        self.assertEqual(engine, "jev")
        self.assertEqual(rec["recommended_agent"], "programmer")
        self.assertEqual(rec["confidence"], "high")
        self.assertEqual(rec["routing_class"], "code_config_work")
        self.assertFalse(rec["needs_identifier"])
        self.assertIn("programmer 0.97", rec["reason"])
        llm.assert_not_awaited()
        state, questions = decide.await_args.args[1], decide.await_args.args[2]
        self.assertEqual(state, {"task": "fix the scheduler"})
        self.assertEqual(set(questions["agent"]["criteria"]), set(tools._DELEGATABLE_AGENTS))
        self.assertIn("political line", questions["agent"]["criteria"]["programmer"])

    async def test_candidates_restrict_the_choice(self):
        with patch.object(cr, "decide", AsyncMock(return_value=decision("scout", 0.9))) as decide:
            rec, engine = await tools._classify_route("check the mailbox", ["scout", "browser"])
        self.assertEqual(engine, "jev")
        self.assertEqual(set(decide.await_args.args[2]["agent"]["criteria"]), {"scout", "browser"})
        self.assertEqual([a["agent"] for a in rec["alternatives"]], ["browser"])

    async def test_low_confidence_falls_back_to_llm_with_hint(self):
        llm_result = {"recommended_agent": "analyst", "confidence": "medium", "reason": "llm", "source": "llm_classifier"}
        with patch.object(cr, "decide", AsyncMock(return_value=decision("analyst", 0.41))), \
             patch.object(tools, "_classify_route_with_llm", AsyncMock(return_value=llm_result)) as llm:
            rec, engine = await tools._classify_route("정치노선 보강", None)
        self.assertEqual(engine, "llm")
        llm.assert_awaited_once()
        self.assertEqual(rec["system_one_hint"], {"recommended_agent": "analyst", "confidence_score": 0.41,
                                                  "routing_class": "code_config_work"})

    async def test_low_confidence_decision_survives_llm_outage(self):
        with patch.object(cr, "decide", AsyncMock(return_value=decision("analyst", 0.6))), \
             patch.object(tools, "_classify_route_with_llm", AsyncMock(return_value=None)):
            rec, engine = await tools._classify_route("x", None)
        self.assertEqual(engine, "jev_low_confidence")
        self.assertEqual(rec["confidence"], "low")
        self.assertNotIn("below_threshold", rec)

    async def test_jev_outage_uses_llm(self):
        llm_result = {"recommended_agent": "diary", "source": "llm_classifier"}
        with patch.object(cr, "decide", AsyncMock(return_value=None)), \
             patch.object(tools, "_classify_route_with_llm", AsyncMock(return_value=llm_result)):
            rec, engine = await tools._classify_route("x", None)
        self.assertEqual((engine, rec["recommended_agent"]), ("llm", "diary"))
        self.assertNotIn("system_one_hint", rec)

    async def test_disabled_feature_skips_jev(self):
        cr.resolve.return_value = cr.CallSiteProfile(feature="t", provider="openrouter", model="m", extra={"enabled": False})
        with patch.object(cr, "decide", AsyncMock()) as decide, \
             patch.object(tools, "_classify_route_with_llm", AsyncMock(return_value=None)):
            self.assertEqual(await tools._classify_route("x", None), (None, None))
        decide.assert_not_awaited()

    async def test_route_task_payload_reports_engine(self):
        with patch.object(cr, "decide", AsyncMock(return_value=decision("browser", 0.93, routing_class="browser_automation"))):
            payload = json.loads(await tools._exec_route_task("log into the site and export the form", include_store_guide=False))
        self.assertEqual(payload["classifier"]["engine"], "jev")
        self.assertTrue(payload["classifier"]["used"])
        self.assertEqual(payload["recommendation"]["recommended_agent"], "browser")


if __name__ == "__main__":
    unittest.main()
