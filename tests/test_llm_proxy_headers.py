"""The proxy forwards client headers minus hop-by-hop, auth and internal ones."""
import unittest

from unittest import mock

from llm_proxy import app as proxy_app
from llm_proxy.app import build_forward_headers, missing_caller_reason


class ForwardHeaderTests(unittest.TestCase):
    def test_internal_caller_and_auth_headers_never_reach_the_provider(self):
        incoming = {"authorization": "Bearer via-llm-proxy", "x-llm-caller": "commulingo_person_classification",
                    "content-type": "application/json", "host": "127.0.0.1:8110", "x-api-key": "old"}
        out = build_forward_headers(incoming, {"auth": ("bearer",)}, "real-key")
        self.assertEqual(out["authorization"], "Bearer real-key")
        self.assertEqual(out["content-type"], "application/json")
        for name in ("x-llm-caller", "host", "x-api-key"):
            self.assertNotIn(name, out)


class CallerRequiredTests(unittest.TestCase):
    def test_unnamed_post_is_refused_and_named_post_passes(self):
        self.assertIn("x-llm-caller", missing_caller_reason("POST", {}))
        self.assertIn("x-llm-caller", missing_caller_reason("POST", {"x-llm-caller": "  "}))
        self.assertIsNone(missing_caller_reason("POST", {"x-llm-caller": "kg_graphiti"}))

    def test_get_passes_unnamed(self):
        self.assertIsNone(missing_caller_reason("GET", {}))

    def test_policy_can_switch_the_requirement_off(self):
        with mock.patch.object(proxy_app, "load_policy", return_value={"require_caller": False}):
            self.assertIsNone(missing_caller_reason("POST", {}))


class ClientStampTests(unittest.TestCase):
    def test_wrapped_sdk_clients_send_the_caller_on_every_request(self):
        import anthropic
        from openai import AsyncOpenAI
        from llm.instrumented_clients import AuditedAsyncAnthropic, AuditedAsyncOpenAI, genai_http_options

        a = AuditedAsyncAnthropic(anthropic.AsyncAnthropic(api_key="k"), caller="site_a", provider="anthropic")
        o = AuditedAsyncOpenAI(AsyncOpenAI(api_key="k"), caller="site_o", provider="openai")
        self.assertEqual(a._client.default_headers["x-llm-caller"], "site_a")
        self.assertEqual(o._client.default_headers["x-llm-caller"], "site_o")
        self.assertEqual(genai_http_options(None, "x"), {})
        self.assertEqual(genai_http_options("http://p/gemini", "x")["http_options"]["headers"],
                         {"x-llm-caller": "x"})


if __name__ == "__main__":
    unittest.main()
