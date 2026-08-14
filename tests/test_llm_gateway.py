"""Hermetic unit tests for llm/gateway.py — the LLM-call seam.

No DB, no API keys: the DB sink is disabled via LENINBOT_LLM_AUDIT_DB=0
(set here defensively as well as in run_unit_tests.sh) and policy/spend
lookups are patched.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ["LENINBOT_LLM_AUDIT_DB"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import llm.gateway as gw  # noqa: E402
from llm.gateway import (  # noqa: E402
    LLMGatewayDenied,
    check_llm_call,
    estimate_cost_usd,
    infer_provider,
    record_llm_call,
)


def _policy(**overrides):
    return {**gw._DEFAULTS, **overrides}


class TestInferProvider(unittest.TestCase):
    def test_known_prefixes(self):
        self.assertEqual(infer_provider("claude-fable-5"), "claude")
        self.assertEqual(infer_provider("gpt-5.6-terra"), "openai")
        self.assertEqual(infer_provider("deepseek-v4-pro"), "deepseek")
        self.assertEqual(infer_provider("kimi-k3"), "kimi")
        self.assertEqual(infer_provider("qwen3.6-9b"), "local")
        self.assertEqual(infer_provider("gemini-3.1-flash-lite"), "gemini")

    def test_unknown_and_empty(self):
        self.assertIsNone(infer_provider("llama-8b"))
        self.assertIsNone(infer_provider(None))
        self.assertIsNone(infer_provider(""))


class TestEstimateCost(unittest.TestCase):
    def test_anthropic_semantics_input_excludes_cache(self):
        # deepseek-v4-flash: in 0.14, out 0.28, cache_read 0.0028 per M
        cost = estimate_cost_usd(
            "deepseek-v4-flash",
            tokens_in=1_000_000, tokens_out=1_000_000, cache_read=1_000_000,
        )
        self.assertAlmostEqual(cost, 0.14 + 0.28 + 0.0028, places=6)

    def test_dated_model_prefix_match(self):
        exact = estimate_cost_usd("claude-haiku-4-5", tokens_in=1000, tokens_out=100)
        dated = estimate_cost_usd(
            "claude-haiku-4-5-20251001", tokens_in=1000, tokens_out=100,
        )
        self.assertIsNotNone(exact)
        self.assertEqual(exact, dated)

    def test_openai_semantics_input_includes_cache(self):
        # gpt-5.6-luna: in 0.20, cached 0.02, out 1.20 per M; prompt_tokens
        # includes the cached share, which must be re-priced, not double-billed.
        cost = estimate_cost_usd(
            "gpt-5.6-luna",
            tokens_in=2_000_000, tokens_out=0, cache_read=1_000_000,
        )
        self.assertAlmostEqual(cost, 0.20 + 0.02, places=6)

    def test_overlapping_model_uses_explicit_protocol_semantics(self):
        openai_cost = estimate_cost_usd(
            "deepseek-v4-flash", tokens_in=2_000_000,
            cache_read=1_000_000, token_semantics="openai",
        )
        anthropic_cost = estimate_cost_usd(
            "deepseek-v4-flash", tokens_in=2_000_000,
            cache_read=1_000_000, token_semantics="anthropic",
        )
        self.assertAlmostEqual(openai_cost, 0.1428, places=6)
        self.assertAlmostEqual(anthropic_cost, 0.2828, places=6)

    def test_gemini_pricing_and_cache_semantics(self):
        cost = estimate_cost_usd(
            "gemini-2.5-flash-lite", tokens_in=2_000_000,
            tokens_out=1_000_000, cache_read=1_000_000,
            token_semantics="gemini",
        )
        self.assertAlmostEqual(cost, 0.10 + 0.01 + 0.40, places=6)

    def test_unknown_model_returns_none_not_fabricated(self):
        self.assertIsNone(estimate_cost_usd("mystery-9000", tokens_in=1000))
        self.assertIsNone(estimate_cost_usd(None, tokens_in=1000))


class TestCheckLLMCall(unittest.TestCase):
    def test_default_policy_allows(self):
        with patch.object(gw, "load_policy", return_value=_policy()):
            with patch.object(gw, "_emit") as emit:
                check_llm_call(surface="loop", caller="analyst", model="claude-fable-5")
        emit.assert_not_called()

    def test_block_all_shadow_records_but_allows(self):
        with patch.object(
            gw, "load_policy", return_value=_policy(block_all=True, enforce=False)
        ):
            with patch.object(gw, "_emit") as emit:
                check_llm_call(surface="loop", caller="analyst", model="claude-fable-5")
        row = emit.call_args[0][0]
        self.assertEqual(row["status"], "would_deny")

    def test_block_all_enforce_raises(self):
        with patch.object(
            gw, "load_policy", return_value=_policy(block_all=True, enforce=True)
        ):
            with patch.object(gw, "_emit"):
                with self.assertRaises(LLMGatewayDenied):
                    check_llm_call(surface="loop", caller="analyst", model="claude-fable-5")

    def test_blocked_provider_inferred_from_model(self):
        pol = _policy(blocked_providers=["openai"], enforce=True)
        with patch.object(gw, "load_policy", return_value=pol):
            with patch.object(gw, "_emit"):
                with self.assertRaises(LLMGatewayDenied):
                    check_llm_call(surface="oneshot", caller="x", model="gpt-5.6-terra")
                # Other providers pass.
                check_llm_call(surface="oneshot", caller="x", model="deepseek-v4-flash")

    def test_daily_budget_enforced(self):
        pol = _policy(daily_budget_usd=10.0, enforce=True)
        with patch.object(gw, "load_policy", return_value=pol):
            with patch.object(gw, "_today_spend", return_value={"claude": 11.5}):
                with patch.object(gw, "_emit"):
                    with self.assertRaises(LLMGatewayDenied):
                        check_llm_call(surface="loop", caller="x", model="claude-fable-5")
            with patch.object(gw, "_today_spend", return_value={"claude": 3.0}):
                check_llm_call(surface="loop", caller="x", model="claude-fable-5")

    def test_per_provider_budget(self):
        pol = _policy(daily_budget_per_provider={"deepseek": 1.0}, enforce=True)
        with patch.object(gw, "load_policy", return_value=pol):
            with patch.object(
                gw, "_today_spend", return_value={"deepseek": 2.0, "claude": 50.0}
            ):
                with patch.object(gw, "_emit"):
                    with self.assertRaises(LLMGatewayDenied):
                        check_llm_call(surface="loop", caller="x", model="deepseek-v4-pro")
                # Uncapped provider unaffected by its own spend.
                check_llm_call(surface="loop", caller="x", model="claude-fable-5")

    def test_budget_fails_open_when_spend_unknown(self):
        pol = _policy(daily_budget_usd=0.01, enforce=True)
        with patch.object(gw, "load_policy", return_value=pol):
            with patch.object(gw, "_today_spend", return_value=None):
                check_llm_call(surface="loop", caller="x", model="claude-fable-5")

    def test_internal_error_fails_open(self):
        with patch.object(gw, "load_policy", side_effect=RuntimeError("boom")):
            check_llm_call(surface="loop", caller="x", model="claude-fable-5")


class TestRecordLLMCall(unittest.TestCase):
    def test_records_with_explicit_cost(self):
        with patch.object(gw, "_emit") as emit:
            record_llm_call(
                surface="loop", caller="analyst", model="claude-fable-5",
                label="Round 1", tokens_in=100, tokens_out=50,
                cache_read=200, cache_create=10, cost_usd=0.0123,
            )
        row = emit.call_args[0][0]
        self.assertEqual(row["provider"], "claude")
        self.assertEqual(row["cost_usd"], 0.0123)
        self.assertEqual(row["status"], "ok")

    def test_cost_autofilled_from_pricing(self):
        with patch.object(gw, "_emit") as emit:
            record_llm_call(
                surface="oneshot", caller="chunk_summary", model="deepseek-v4-flash",
                tokens_in=1_000_000, tokens_out=0,
            )
        self.assertAlmostEqual(emit.call_args[0][0]["cost_usd"], 0.14, places=6)

    def test_never_raises(self):
        with patch.object(gw, "estimate_cost_usd", side_effect=RuntimeError("boom")):
            record_llm_call(surface="loop", caller="x", model="m")  # must not raise

    def test_error_excerpt_truncated(self):
        with patch.object(gw, "_emit") as emit:
            record_llm_call(
                surface="oneshot", caller="x", model="deepseek-v4-flash",
                status="error", error_excerpt="e" * 2000,
            )
        self.assertLessEqual(len(emit.call_args[0][0]["error_excerpt"]), 510)


class TestLoopStateSeam(unittest.TestCase):
    """LoopState.add_cost is the loop-side seam: it must forward usage."""

    def test_add_cost_forwards_to_gateway(self):
        import llm.agent_loop as agent_loop

        with patch.object(agent_loop, "record_llm_call") as rec:
            state = agent_loop.LoopState(0.5, agent_name="analyst")
            state.add_cost(
                0.01, model="deepseek-v4-pro", label="Round 3",
                tokens_in=1000, tokens_out=200, cache_read=5000, cache_create=0,
            )
        self.assertAlmostEqual(state.total_cost, 0.01)
        kwargs = rec.call_args.kwargs
        self.assertEqual(kwargs["surface"], "loop")
        self.assertEqual(kwargs["caller"], "analyst")
        self.assertEqual(kwargs["model"], "deepseek-v4-pro")
        self.assertEqual(kwargs["cost_usd"], 0.01)
        self.assertEqual(kwargs["cache_read"], 5000)

    def test_add_cost_bare_still_works(self):
        # Compatibility: cost-only calls (older call shape) must still count.
        import llm.agent_loop as agent_loop

        with patch.object(agent_loop, "record_llm_call"):
            state = agent_loop.LoopState(0.5)
            state.add_cost(0.02)
            state.add_cost(0.03)
        self.assertAlmostEqual(state.total_cost, 0.05)


class TestProxyRouting(unittest.TestCase):
    def test_direct_mode_passthrough(self):
        with patch.object(gw, "load_policy", return_value=_policy()):
            base, key = gw.provider_endpoint(
                "deepseek/anthropic", "https://api.deepseek.com/anthropic", "realkey",
            )
        self.assertEqual(base, "https://api.deepseek.com/anthropic")
        self.assertEqual(key, "realkey")

    def test_direct_mode_missing_key_stays_empty(self):
        with patch.object(gw, "load_policy", return_value=_policy()):
            _base, key = gw.provider_endpoint("anthropic", None, "")
        self.assertEqual(key, "")

    def test_proxy_mode_routes_and_placeholder(self):
        pol = _policy(proxy_base="http://127.0.0.1:8100")
        with patch.object(gw, "load_policy", return_value=pol):
            base, key = gw.provider_endpoint("moonshot/v1", "https://api.moonshot.ai/v1", "")
        self.assertEqual(base, "http://127.0.0.1:8100/moonshot/v1")
        self.assertEqual(key, gw.PROXY_PLACEHOLDER_KEY)

    def test_proxy_mode_keeps_real_key_when_present(self):
        pol = _policy(proxy_base="http://127.0.0.1:8100/")
        with patch.object(gw, "load_policy", return_value=pol):
            base, key = gw.provider_endpoint("anthropic", None, "realkey")
        self.assertEqual(base, "http://127.0.0.1:8100/anthropic")
        self.assertEqual(key, "realkey")


class TestRegistryProviderConnections(unittest.TestCase):
    def test_proxy_mode_resolves_route_and_placeholder_together(self):
        from llm import call_registry

        pol = _policy(proxy_base="http://127.0.0.1:8110")
        with patch.object(gw, "load_policy", return_value=pol):
            with patch.object(call_registry, "get_secret", return_value=""):
                with patch.dict(os.environ, {"DEEPSEEK_ANTHROPIC_BASE_URL": ""}):
                    connection = call_registry.resolve_provider_connection(
                        "deepseek_anthropic"
                    )
        self.assertEqual(
            connection.base_url, "http://127.0.0.1:8110/deepseek/anthropic"
        )
        self.assertEqual(connection.api_key, gw.PROXY_PLACEHOLDER_KEY)
        self.assertEqual(connection.credential_name, "DEEPSEEK_API_KEY")

    def test_direct_mode_requires_real_key(self):
        from llm import call_registry

        with patch.object(gw, "load_policy", return_value=_policy()):
            with patch.object(call_registry, "get_secret", return_value=""):
                with self.assertRaises(call_registry.ProviderConnectionError) as raised:
                    call_registry.resolve_provider_connection("gemini")
        self.assertEqual(raised.exception.credential_name, "GEMINI_API_KEY")

    def test_explicit_direct_base_does_not_receive_proxy_placeholder(self):
        from llm import call_registry

        pol = _policy(proxy_base="http://127.0.0.1:8110")
        with patch.object(gw, "load_policy", return_value=pol):
            with patch.object(call_registry, "get_secret", return_value=""):
                with patch.dict(
                    os.environ,
                    {"DEEPSEEK_ANTHROPIC_BASE_URL": "https://override.invalid/api/"},
                ):
                    with self.assertRaises(call_registry.ProviderConnectionError):
                        call_registry.resolve_provider_connection("deepseek_anthropic")

    def test_archival_preflight_accepts_keyless_gateway_mode(self):
        from llm import call_registry
        from runtime_tools.archival_translation import core

        pol = _policy(proxy_base="http://127.0.0.1:8110")
        with patch.object(gw, "load_policy", return_value=pol):
            with patch.object(call_registry, "get_secret", return_value=""):
                with patch.dict(os.environ, {"DEEPSEEK_ANTHROPIC_BASE_URL": ""}):
                    core.preflight()

    def test_archival_preflight_rejects_missing_direct_key(self):
        from llm import call_registry
        from runtime_tools.archival_translation import core

        with patch.object(gw, "load_policy", return_value=_policy()):
            with patch.object(call_registry, "get_secret", return_value=""):
                with patch.dict(os.environ, {"DEEPSEEK_ANTHROPIC_BASE_URL": ""}):
                    with self.assertRaises(core.SpecError) as raised:
                        core.preflight()
        self.assertIn("DEEPSEEK_API_KEY", str(raised.exception))


class TestEvaluatePolicy(unittest.TestCase):
    """The shared decision function both evaluation points call."""

    def test_allow_returns_none_with_enforce_posture(self):
        with patch.object(gw, "load_policy", return_value=_policy()):
            reason, enforce = gw.evaluate_policy(provider="claude", model="claude-fable-5")
        self.assertIsNone(reason)
        self.assertTrue(enforce)

    def test_deny_reason_with_enforce_flag(self):
        pol = _policy(blocked_providers=["kimi"], enforce=True)
        with patch.object(gw, "load_policy", return_value=pol):
            reason, enforce = gw.evaluate_policy(provider="kimi", model="kimi-k3")
        self.assertIn("kimi", reason)
        self.assertTrue(enforce)

    def test_never_raises(self):
        with patch.object(gw, "load_policy", side_effect=RuntimeError("boom")):
            reason, enforce = gw.evaluate_policy(provider="claude", model=None)
        self.assertIsNone(reason)
        self.assertFalse(enforce)


class TestProxyPolicyGate(unittest.TestCase):
    def test_model_from_body(self):
        from llm_proxy.app import model_from_body

        self.assertEqual(
            model_from_body(b'{"model": "deepseek-v4-flash", "messages": []}'),
            "deepseek-v4-flash",
        )
        self.assertIsNone(model_from_body(b""))
        self.assertIsNone(model_from_body(b"not json"))
        self.assertIsNone(model_from_body(b'{"messages": []}'))

    def test_gemini_model_from_url_path(self):
        from llm_proxy.app import model_from_request

        self.assertEqual(
            model_from_request(
                "gemini", "v1beta/models/gemini-2.5-flash-lite%3AgenerateContent", b"{}",
            ),
            "gemini-2.5-flash-lite",
        )
        self.assertEqual(
            model_from_request(
                "gemini", "v1beta/models/gemini-embedding-001:embedContent", b"",
            ),
            "gemini-embedding-001",
        )

    def test_route_to_policy_provider_mapping(self):
        from llm_proxy.app import POLICY_PROVIDER, PROVIDERS

        self.assertEqual(POLICY_PROVIDER["anthropic"], "claude")
        self.assertEqual(POLICY_PROVIDER["moonshot"], "kimi")
        # Unmapped routes are their own policy names.
        for route in PROVIDERS:
            name = POLICY_PROVIDER.get(route, route)
            self.assertIn(name, {"claude", "kimi", "deepseek", "openai", "gemini"})

    def test_each_proxy_provider_has_exactly_one_credential(self):
        from llm_proxy.app import PROVIDERS

        for cfg in PROVIDERS.values():
            self.assertIn("secret", cfg)
            self.assertNotIn("secrets", cfg)
            self.assertIsInstance(cfg["secret"], str)


class TestProxyHeaderInjection(unittest.TestCase):
    """llm_proxy must strip client auth and inject the real key, untouched rest."""

    def test_anthropic_style(self):
        from llm_proxy.app import PROVIDERS, build_forward_headers

        headers = build_forward_headers(
            {
                "x-api-key": "via-llm-proxy", "anthropic-version": "2023-06-01",
                "content-type": "application/json", "host": "127.0.0.1:8100",
                "content-length": "42",
            },
            PROVIDERS["anthropic"], "REALKEY",
        )
        self.assertEqual(headers["x-api-key"], "REALKEY")
        self.assertEqual(headers["anthropic-version"], "2023-06-01")
        self.assertNotIn("host", headers)
        self.assertNotIn("content-length", headers)

    def test_bearer_style_replaces_placeholder(self):
        from llm_proxy.app import PROVIDERS, build_forward_headers

        headers = build_forward_headers(
            {"authorization": "Bearer via-llm-proxy", "content-type": "application/json"},
            PROVIDERS["moonshot"], "REALKEY",
        )
        self.assertEqual(headers["authorization"], "Bearer REALKEY")

    def test_deepseek_gets_both_auth_styles(self):
        from llm_proxy.app import PROVIDERS, build_forward_headers

        headers = build_forward_headers({}, PROVIDERS["deepseek"], "K")
        self.assertEqual(headers["x-api-key"], "K")
        self.assertEqual(headers["authorization"], "Bearer K")


class _FakeUpstream:
    """Stand-in for httpx.Response: a chunk script, optional mid-stream error."""

    def __init__(self, chunks, status_code=200, raise_after=None):
        self.status_code = status_code
        self.closed = False
        self._chunks = chunks
        self._raise_after = raise_after

    async def aiter_raw(self):
        for chunk in self._chunks:
            yield chunk
        if self._raise_after is not None:
            raise self._raise_after

    async def aclose(self):
        self.closed = True


class TestProxyStreamAudit(unittest.TestCase):
    """The proxy transport row must reflect the actual STREAM outcome."""

    def _run(self, coro):
        import asyncio

        return asyncio.run(coro)

    def _relay(self, upstream):
        from llm_proxy.app import relay_and_record

        return relay_and_record(
            upstream, caller="tester", provider="claude", model="claude-x",
            label="v1/messages", started=0.0,
        )

    def test_completed_stream_records_ok_once(self):
        upstream = _FakeUpstream([b"a", b"bc"])
        with patch("llm_proxy.app.record_llm_call") as rec:
            async def consume():
                return [c async for c in self._relay(upstream)]

            chunks = self._run(consume())
        self.assertEqual(chunks, [b"a", b"bc"])
        self.assertTrue(upstream.closed)
        self.assertEqual(rec.call_count, 1)
        kwargs = rec.call_args.kwargs
        self.assertEqual(kwargs["status"], "ok")
        self.assertIsNone(kwargs["error_excerpt"])
        self.assertFalse(kwargs["estimate_cost"])

    def test_upstream_http_error_status_recorded_after_body(self):
        upstream = _FakeUpstream([b'{"error":1}'], status_code=429)
        with patch("llm_proxy.app.record_llm_call") as rec:
            async def consume():
                return [c async for c in self._relay(upstream)]

            self._run(consume())
        kwargs = rec.call_args.kwargs
        self.assertEqual(kwargs["status"], "error")
        self.assertIn("HTTP 429", kwargs["error_excerpt"])

    def test_mid_stream_abort_records_error(self):
        upstream = _FakeUpstream([b"a"], raise_after=RuntimeError("conn reset"))
        with patch("llm_proxy.app.record_llm_call") as rec:
            async def consume():
                async for _ in self._relay(upstream):
                    pass

            with self.assertRaises(RuntimeError):
                self._run(consume())
        self.assertTrue(upstream.closed)
        kwargs = rec.call_args.kwargs
        self.assertEqual(kwargs["status"], "error")
        self.assertIn("stream aborted: RuntimeError: conn reset", kwargs["error_excerpt"])

    def test_client_disconnect_records_error(self):
        upstream = _FakeUpstream([b"a", b"b", b"c"])
        with patch("llm_proxy.app.record_llm_call") as rec:
            async def disconnect_after_first():
                gen = self._relay(upstream)
                await gen.__anext__()
                await gen.aclose()  # what starlette does when the client goes away

            self._run(disconnect_after_first())
        self.assertTrue(upstream.closed)
        self.assertEqual(rec.call_count, 1)
        kwargs = rec.call_args.kwargs
        self.assertEqual(kwargs["status"], "error")
        self.assertIn("client disconnected", kwargs["error_excerpt"])


class TestPolicyLoading(unittest.TestCase):
    def setUp(self):
        gw._config_cache = None
        gw._config_signature = None

    def tearDown(self):
        gw._config_cache = None
        gw._config_signature = None

    def test_missing_config_uses_defaults(self):
        with patch.object(
            gw, "DEFAULT_CONFIG_PATH", Path("/nonexistent/llm_gateway.defaults.json")
        ), patch.object(
            gw, "LOCAL_CONFIG_PATH", Path("/nonexistent/llm_gateway.local.json")
        ):
            pol = gw.load_policy()
        self.assertTrue(pol["enforce"])
        self.assertFalse(pol["block_all"])

    def test_repo_defaults_are_safe_and_enforced(self):
        pol = gw.load_policy()
        self.assertTrue(pol["enforce"])
        self.assertFalse(pol["block_all"], "repo default must not block calls")
        self.assertEqual(pol["proxy_base"], "http://127.0.0.1:8110")

    def test_local_overrides_take_precedence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            defaults = root / "llm_gateway.defaults.json"
            local = root / "llm_gateway.local.json"
            defaults.write_text(
                json.dumps({"enforce": True, "daily_budget_usd": None}),
                encoding="utf-8",
            )
            local.write_text(
                json.dumps({"enforce": False, "daily_budget_usd": 12.5}),
                encoding="utf-8",
            )
            with patch.object(gw, "DEFAULT_CONFIG_PATH", defaults), patch.object(
                gw, "LOCAL_CONFIG_PATH", local
            ):
                pol = gw.load_policy()
        self.assertFalse(pol["enforce"])
        self.assertEqual(pol["daily_budget_usd"], 12.5)

    def test_invalid_local_config_keeps_last_good_policy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            defaults = root / "llm_gateway.defaults.json"
            local = root / "llm_gateway.local.json"
            defaults.write_text(json.dumps({"enforce": True}), encoding="utf-8")
            local.write_text(json.dumps({"block_all": False}), encoding="utf-8")
            with patch.object(gw, "DEFAULT_CONFIG_PATH", defaults), patch.object(
                gw, "LOCAL_CONFIG_PATH", local
            ):
                first = gw.load_policy()
                local.write_text("{broken", encoding="utf-8")
                second = gw.load_policy()
        self.assertIs(first, second)
        self.assertFalse(second["block_all"])


class TestGatewayCLIOverrides(unittest.TestCase):
    def test_set_writes_only_local_override(self):
        from scripts import llm_gateway_cli as cli

        with tempfile.TemporaryDirectory() as tmpdir:
            local = Path(tmpdir) / "llm_gateway.local.json"
            args = SimpleNamespace(key="daily_budget_usd", value="25")
            with patch.object(cli, "LOCAL_CONFIG_PATH", local), patch("builtins.print"):
                self.assertEqual(cli.cmd_set(args), 0)
            self.assertEqual(
                json.loads(local.read_text(encoding="utf-8")),
                {"daily_budget_usd": 25},
            )
            self.assertFalse(list(local.parent.glob(f".{local.name}.*.tmp")))

    def test_unset_restores_inheritance(self):
        from scripts import llm_gateway_cli as cli

        with tempfile.TemporaryDirectory() as tmpdir:
            local = Path(tmpdir) / "llm_gateway.local.json"
            local.write_text(
                json.dumps({"enforce": False, "block_all": True}),
                encoding="utf-8",
            )
            args = SimpleNamespace(key="enforce")
            with patch.object(cli, "LOCAL_CONFIG_PATH", local), patch("builtins.print"):
                self.assertEqual(cli.cmd_unset(args), 0)
            self.assertEqual(
                json.loads(local.read_text(encoding="utf-8")),
                {"block_all": True},
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
