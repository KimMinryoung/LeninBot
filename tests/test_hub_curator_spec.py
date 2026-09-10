"""Sanity checks for the hub_curator AgentSpec and its prompt."""

import re
import unittest

from agents.commulingo_curator import EDITORIAL_CORE
from agents.hub_curator import CURATION_LIMITS, HUB_CURATOR
from security_gateway.policy import TOOL_RISK_CLASS, risk_class


class HubCuratorSpecTests(unittest.TestCase):
    def test_single_terminal_publish_tool(self):
        self.assertEqual(HUB_CURATOR.terminal_tools, ["publish_hub_curation"])
        self.assertEqual(HUB_CURATOR.finalization_tools, ["publish_hub_curation"])
        self.assertIn("publish_hub_curation", HUB_CURATOR.tools)
        self.assertIn("fetch_url", HUB_CURATOR.tools)

    def test_provider_and_model(self):
        self.assertEqual(HUB_CURATOR.provider, "deepseek")
        self.assertEqual(HUB_CURATOR.model, "deepseek_flash")
        self.assertTrue(HUB_CURATOR.skip_orchestrator_report)

    def test_every_tool_has_a_gateway_risk_class(self):
        # Uncategorized tools are denied by the security gateway; every tool the
        # spec exposes must be classified so the run cannot stall on a denial.
        for tool in HUB_CURATOR.tools:
            self.assertIn(tool, TOOL_RISK_CLASS, tool)
        self.assertEqual(risk_class("publish_hub_curation"), "publish")

    def test_prompt_is_complete(self):
        identity = HUB_CURATOR.prompt_ir.identity
        self.assertNotIn("__", identity)
        self.assertIn(EDITORIAL_CORE, identity)
        self.assertRegex(identity, r"[가-힣]")
        self.assertIn(str(CURATION_LIMITS["rationale_chars"][0]), identity)
        self.assertIn(str(CURATION_LIMITS["context_chars"][1]), identity)
        self.assertIn(str(CURATION_LIMITS["slug_max"]), identity)

    def test_registered_and_runtime_config_applies(self):
        import agents

        spec = agents.get_agent("hub_curator")
        self.assertIs(spec, HUB_CURATOR)
        self.assertEqual(spec.terminal_tools, ["publish_hub_curation"])
        self.assertTrue(set(spec.terminal_tools) <= set(spec.tools))


if __name__ == "__main__":
    unittest.main()
