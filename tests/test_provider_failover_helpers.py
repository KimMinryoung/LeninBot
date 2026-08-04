"""Unit tests for the shared provider-dispatch helpers in llm.provider_failover.

resolve_runtime_profile is patched — no config/DB access.
Run from repo root:  venv/bin/python -m unittest discover tests -v
"""
import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import runtime_profile
from llm.provider_failover import resolve_deepseek_failover_model
from llm.provider_registry import kimi_openai_tool_options


def _fake_profile(model_id):
    async def _resolve(kind, **kwargs):
        return SimpleNamespace(model_id=model_id, requested=(kind, kwargs))
    return _resolve


class TestKimiToolOptions(unittest.TestCase):
    def test_contract_has_no_content_filter_fallback(self):
        # The Kimi→DeepSeek content-filter switching machinery was removed
        # 2026-08-04; the contract is now static provider options only.
        opts = kimi_openai_tool_options()
        self.assertNotIn("content_filter_fallback_client", opts)
        self.assertEqual(opts["sdk_max_token_param"], "max_tokens")
        self.assertFalse(opts["include_parallel_tool_calls"])
        self.assertTrue(opts["preserve_reasoning_content"])


class TestDeepseekFailoverModel(unittest.TestCase):
    def test_no_openai_client_disables_failover(self):
        self.assertIsNone(asyncio.run(resolve_deepseek_failover_model("chat", None)))

    def test_resolves_openai_medium_tier(self):
        with patch.object(runtime_profile, "resolve_runtime_profile",
                          _fake_profile("gpt-5.6-terra")):
            model = asyncio.run(resolve_deepseek_failover_model("task", object()))
        self.assertEqual(model, "gpt-5.6-terra")

    def test_resolution_failure_returns_none(self):
        async def _boom(kind, **kwargs):
            raise RuntimeError("config unavailable")

        with patch.object(runtime_profile, "resolve_runtime_profile", _boom):
            model = asyncio.run(resolve_deepseek_failover_model("chat", object()))
        self.assertIsNone(model)


if __name__ == "__main__":
    unittest.main()
