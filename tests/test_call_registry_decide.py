"""System One (Jev) decisions: typed answers, audit rows, and fallbacks."""
import unittest
from unittest import mock

from llm import call_registry as cr
from llm import gateway


def _profile(provider="openrouter", model="typesafe/jev-1.13"):
    return cr.CallSiteProfile(feature="t", provider=provider, model=model, timeout=5.0)


QUESTIONS = {
    "urgent": {"type": "noul", "instructions": "The message conveys urgency."},
    "dept": {"type": "choice", "instructions": "Which team?",
             "criteria": {"billing": "money", "tech": {"nested": True}}},
    "mood": {"type": "score", "instructions": "How angry?", "criteria": ["calm", "annoyed", "furious"]},
}

PAYLOAD = {
    "model": "typesafe/jev-1.13-20260917",
    "answers": {
        "urgent": {"type": "noul", "noul": 0.83},
        "dept": {"type": "choice", "choice": "billing",
                 "probabilities": {"billing": 0.9, "tech": 0.1}, "confidence": 0.88},
        "mood": {"type": "score", "score": 1.4, "legend": {"0": "calm"},
                 "probabilities": {"calm": 0.1, "annoyed": 0.5, "furious": 0.4}, "confidence": 0.5},
    },
    "usage": {"input_tokens": 501, "output_tokens": 89, "cost": 2.1042e-05},
}


class _Resp:
    def __init__(self, status_code, payload=None, text="", headers=None):
        self.status_code, self._payload, self.text = status_code, payload, text
        self.headers = headers or {}

    def json(self):
        return self._payload


class _DecideCase(unittest.TestCase):
    def setUp(self):
        self.recorded = []
        patches = [
            mock.patch.object(gateway, "record_llm_call", side_effect=lambda **kw: self.recorded.append(kw)),
            mock.patch.object(gateway, "check_llm_call", return_value=None),
            mock.patch.object(cr, "resolve_provider_connection", side_effect=lambda provider: cr.ProviderConnection(
                provider=provider, credential_name=f"{provider.upper()}_API_KEY",
                base_url=f"http://127.0.0.1:8110/{provider}", api_key="via-llm-proxy")),
            # The exhaustion breaker lives in Redis; keep it in memory here.
            mock.patch.object(cr, "_decision_route_exhausted", side_effect=lambda provider: provider in self.exhausted),
            mock.patch.object(cr, "_mark_decision_route_exhausted",
                              side_effect=lambda provider, *a: self.exhausted.add(provider)),
        ]
        self.exhausted = set()
        for p in patches:
            p.start(); self.addCleanup(p.stop)


class DecideTests(_DecideCase):
    def test_success_returns_typed_answers_and_audits_reported_cost(self):
        with mock.patch("httpx.post", return_value=_Resp(200, PAYLOAD)) as post:
            result = cr.decide_detailed("t", {"msg": "help"}, QUESTIONS, profile=_profile(), label="smoke")
        d = result.decision
        self.assertIsNotNone(d)
        self.assertEqual(d.noul("urgent"), 0.83)
        self.assertEqual(d.choice("dept"), "billing")
        self.assertEqual(d.confidence("dept"), 0.88)
        self.assertEqual(d.score("mood"), 1.4)
        self.assertEqual(d.probabilities("mood")["annoyed"], 0.5)
        self.assertIsNone(d.noul("missing"))
        self.assertEqual(d.model, "typesafe/jev-1.13-20260917")
        self.assertAlmostEqual(d.cost_usd, 2.1042e-05)
        # Request shape: decisions path on the resolved base, caller header, nested criteria stringified.
        url = post.call_args.args[0]
        self.assertEqual(url, "http://127.0.0.1:8110/openrouter/api/alpha/decisions")
        body = post.call_args.kwargs["json"]
        self.assertEqual(body["model"], "typesafe/jev-1.13")
        self.assertEqual(body["questions"]["dept"]["criteria"]["tech"], '{"nested": true}')
        self.assertEqual(post.call_args.kwargs["headers"]["x-llm-caller"], "t")
        # One billed audit row with the provider-reported cost and served model.
        self.assertEqual(len(self.recorded), 1)
        row = self.recorded[0]
        self.assertEqual((row["provider"], row["model"], row["tokens_in"], row["label"]),
                         ("openrouter", "typesafe/jev-1.13-20260917", 501, "smoke"))
        self.assertAlmostEqual(row["cost_usd"], 2.1042e-05)

    def test_direct_typesafe_path_and_estimated_cost(self):
        payload = {**PAYLOAD, "model": "jev-1.13.0", "usage": {"input_tokens": 1000, "output_tokens": 0}}
        with mock.patch.object(cr, "resolve_provider_connection", return_value=cr.ProviderConnection(
                provider="typesafe", credential_name="TYPESAFE_API_KEY",
                base_url="https://api.typesafe.ai", api_key="sk")), \
             mock.patch("httpx.post", return_value=_Resp(200, payload)) as post:
            result = cr.decide_detailed("t", "state", QUESTIONS, profile=_profile("typesafe", "jev-1.13.0"))
        self.assertEqual(post.call_args.args[0], "https://api.typesafe.ai/v1/systemone")
        self.assertAlmostEqual(result.decision.cost_usd, 1000 * 0.042 / 1_000_000)

    def test_http_error_is_classified_and_audited_without_raising(self):
        with mock.patch("httpx.post", return_value=_Resp(401, text="Unauthorized")):
            result = cr.decide_detailed("t", "s", QUESTIONS, profile=_profile())
        self.assertIsNone(result.decision)
        self.assertEqual(result.error_kind, "authentication")
        self.assertEqual(self.recorded[0]["status"], "error")
        self.assertEqual(self.exhausted, {"openrouter"})  # an unusable key flags its route
        self.exhausted.clear()
        with mock.patch("httpx.post", return_value=_Resp(429, text="slow down", headers={"retry-after": "7"})):
            result = cr.decide_detailed("t", "s", QUESTIONS, profile=_profile())
        self.assertTrue(result.retryable)
        self.assertEqual(result.retry_after, 7.0)
        with mock.patch("httpx.post", return_value=_Resp(429, text="slow down", headers={"retry-after": "Sat, 1 Jan"})):
            self.assertIsNone(cr.decide_detailed("t", "s", QUESTIONS, profile=_profile()).retry_after)

    def test_non_system_one_profile_is_a_configuration_error(self):
        result = cr.decide_detailed("t", "s", QUESTIONS, profile=_profile("deepseek", "deepseek-flash"))
        self.assertEqual(result.error_kind, "configuration")
        self.assertEqual(self.recorded, [])

    def test_malformed_questions_rejected_before_any_call(self):
        bad = {"x": {"type": "choice", "instructions": "?", "criteria": {"only": "one"}}}
        with mock.patch("httpx.post") as post:
            result = cr.decide_detailed("t", "s", bad, profile=_profile())
        self.assertEqual(result.error_kind, "configuration")
        post.assert_not_called()

    def test_structured_instructions_work_on_wire_and_in_fan_out(self):
        question = {'type': 'noul', 'instructions': {'question': 'Is this urgent?', 'hint': ['deadline']}}
        with mock.patch('httpx.post', return_value=_Resp(200, PAYLOAD)) as post:
            cr.decide_detailed('t', 's', {'urgent': question}, profile=_profile())
        import json
        self.assertEqual(json.loads(post.call_args.kwargs['json']['questions']['urgent']['instructions']),
                         question['instructions'])
        _, questions = cr.fan_out({'c1': 's'}, {'urgent': question})
        self.assertIn('About `items.c1`:', questions['c1_urgent']['instructions'])
        self.assertIn('deadline', questions['c1_urgent']['instructions'])
        self.assertIsInstance(question['instructions'], dict)

    def test_retryable_failure_is_retried_once_then_succeeds(self):
        responses = [_Resp(429, text="slow down", headers={"retry-after": "1"}), _Resp(200, PAYLOAD)]
        with mock.patch("httpx.post", side_effect=responses) as post, \
             mock.patch.object(cr.time, "sleep") as sleep:
            result = cr.decide_detailed("t", "s", QUESTIONS, profile=_profile())
        self.assertIsNotNone(result.decision)
        self.assertEqual(post.call_count, 2)
        sleep.assert_called_once_with(1.0)
        # One error row for the failed attempt, one billed row for the answer.
        self.assertEqual([r.get("status", "ok") for r in self.recorded], ["error", "ok"])
        self.assertEqual(self.recorded[1]["tokens_in"], 501)

    def test_retry_pause_is_capped_and_non_retryable_errors_are_not_retried(self):
        responses = [_Resp(503, text="busy", headers={"retry-after": "30"}), _Resp(200, PAYLOAD)]
        with mock.patch("httpx.post", side_effect=responses), mock.patch.object(cr.time, "sleep") as sleep:
            self.assertIsNotNone(cr.decide_detailed("t", "s", QUESTIONS, profile=_profile()).decision)
        sleep.assert_called_once_with(cr._DECISION_RETRY_PAUSE_MAX)
        with mock.patch("httpx.post", return_value=_Resp(401, text="Unauthorized")) as post, \
             mock.patch.object(cr.time, "sleep") as sleep:
            self.assertEqual(cr.decide_detailed("t", "s", QUESTIONS, profile=_profile()).error_kind, "authentication")
        self.assertEqual(post.call_count, 1)
        sleep.assert_not_called()

    def test_read_timeout_is_not_retried_but_connection_errors_are(self):
        import httpx
        with mock.patch("httpx.post", side_effect=httpx.ReadTimeout("slow")) as post, \
             mock.patch.object(cr.time, "sleep") as sleep:
            result = cr.decide_detailed("t", "s", QUESTIONS, profile=_profile())
        self.assertEqual((result.error_kind, post.call_count), ("transport", 1))
        sleep.assert_not_called()
        with mock.patch("httpx.post", side_effect=[httpx.ConnectError("refused"), _Resp(200, PAYLOAD)]) as post, \
             mock.patch.object(cr.time, "sleep"):
            self.assertIsNotNone(cr.decide_detailed("t", "s", QUESTIONS, profile=_profile()).decision)
        self.assertEqual(post.call_count, 2)

    def test_attempts_follow_the_entry_and_survive_bad_values(self):
        self.assertEqual(cr._decision_attempts(_profile()), 2)
        self.assertEqual(cr._decision_attempts(cr.CallSiteProfile(feature="t", provider="openrouter", model="m",
                                                                  extra={"retries": 3})), 4)
        self.assertEqual(cr._decision_attempts(cr.CallSiteProfile(feature="t", provider="openrouter", model="m",
                                                                  extra={"retries": "one"})), 2)
        self.assertEqual(cr._decision_attempts(cr.CallSiteProfile(feature="t", provider="openrouter", model="m",
                                                                  extra={"retries": None})), 2)

    def test_registry_entry_can_disable_retries(self):
        profile = cr.CallSiteProfile(feature="t", provider="openrouter", model="typesafe/jev-1.13",
                                     timeout=5.0, extra={"retries": 0})
        with mock.patch("httpx.post", return_value=_Resp(429, text="slow down")) as post:
            self.assertTrue(cr.decide_detailed("t", "s", QUESTIONS, profile=profile).retryable)
        self.assertEqual(post.call_count, 1)

    def test_decide_sync_returns_none_on_failure(self):
        with mock.patch("httpx.post", side_effect=ConnectionError("down")):
            self.assertIsNone(cr.decide_sync("t", "s", QUESTIONS, profile=_profile()))


class PricingTests(unittest.TestCase):
    def test_jev_ids_on_every_route_price_input_only(self):
        for model in ("jev-1.13.0", "typesafe/jev-1.13", "typesafe/jev-1.13-20260917", "jev-latest"):
            self.assertAlmostEqual(gateway.estimate_cost_usd(model, tokens_in=1_000_000, tokens_out=5000), 0.042)

    def test_provider_inference(self):
        self.assertEqual(gateway.infer_provider("typesafe/jev-1.13"), "openrouter")
        self.assertEqual(gateway.infer_provider("jev-1.13.0"), "typesafe")
        self.assertEqual(gateway.infer_provider("gemini-3.7-flash"), "gemini")

class DecisionFailoverTests(_DecideCase):
    """TypeSafe credits or key failing: flag the route, retry on OpenRouter."""

    def run_detailed(self, responses):
        calls = []

        def post(url, **kwargs):
            calls.append((url, kwargs["json"]["model"]))
            return responses.pop(0)
        with mock.patch("httpx.post", side_effect=post):
            result = cr.decide_detailed("t", "s", QUESTIONS, profile=_profile("typesafe", "jev-1.13.0"))
        return result, calls

    def test_exhausted_typesafe_fails_over_to_openrouter_with_its_model_id(self):
        for status, text in [(402, "payment required"), (401, "invalid key"), (403, "credit balance too low")]:
            with self.subTest(status=status):
                self.exhausted.clear(); self.recorded.clear()
                result, calls = self.run_detailed([_Resp(status, text=text), _Resp(200, PAYLOAD)])
                self.assertIsNotNone(result.decision)
                self.assertEqual(calls, [("http://127.0.0.1:8110/typesafe/v1/systemone", "jev-1.13.0"),
                                         ("http://127.0.0.1:8110/openrouter/api/alpha/decisions", "typesafe/jev-1.13")])
                self.assertEqual(self.exhausted, {"typesafe"})
                self.assertEqual(self.recorded[-1]["provider"], "openrouter")

    def test_flagged_route_is_skipped_without_a_request(self):
        self.exhausted.add("typesafe")
        result, calls = self.run_detailed([_Resp(200, PAYLOAD)])
        self.assertIsNotNone(result.decision)
        self.assertEqual([url.rsplit("/", 1)[-1] for url, _ in calls], ["decisions"])

    def test_both_routes_exhausted_returns_the_failure_without_requests(self):
        self.exhausted.update({"typesafe", "openrouter"})
        result, calls = self.run_detailed([])
        self.assertIsNone(result.decision)
        self.assertEqual(result.error_kind, "exhausted")
        self.assertEqual(calls, [])

    def test_proxy_policy_denial_neither_flags_nor_fails_over(self):
        # A budget cap must not be sidestepped by switching routes.
        result, calls = self.run_detailed([_Resp(403, text='{"error": "llm gateway policy: daily cap reached"}')])
        self.assertEqual(result.error_kind, "policy")
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.exhausted, set())

    def test_transient_errors_do_not_fail_over_or_flag(self):
        result, calls = self.run_detailed([_Resp(429, text="slow"), _Resp(503, text="down")])
        self.assertIsNone(result.decision)
        self.assertEqual({url.rsplit("/", 1)[-1] for url, _ in calls}, {"systemone"})
        self.assertEqual(self.exhausted, set())

    def test_failover_model_spelling(self):
        self.assertEqual(cr._failover_model("openrouter", "jev-1.13.0"), "typesafe/jev-1.13")
        self.assertEqual(cr._failover_model("openrouter", "typesafe/jev-1.13"), "typesafe/jev-1.13")
        self.assertIsNone(cr._failover_model("openrouter", "other-model"))


class ExhaustionFlagTests(unittest.TestCase):
    def test_first_flag_alerts_owner_once_and_expires(self):
        class FakeRedis:
            def __init__(self):
                self.keys = {}
            def set(self, key, value, nx=False, ex=None):
                if nx and key in self.keys:
                    return None
                self.keys[key] = (value, ex)
                return True
            def exists(self, key):
                return int(key in self.keys)
        fake, alerts = FakeRedis(), []
        with mock.patch("memory_store.redis_state.get_redis", return_value=fake), \
             mock.patch("memory_store.redis_state.push_owner_alert", side_effect=alerts.append):
            cr._mark_decision_route_exhausted("typesafe", "t", "quota", "HTTP 402: payment required")
            cr._mark_decision_route_exhausted("typesafe", "t", "quota", "HTTP 402: payment required")
            self.assertTrue(cr._decision_route_exhausted("typesafe"))
            self.assertFalse(cr._decision_route_exhausted("openrouter"))
        self.assertEqual(fake.keys["jev:exhausted:typesafe"], ("quota", cr._EXHAUSTED_TTL))
        self.assertEqual(len(alerts), 1)
        self.assertIn("openrouter 경로로 우회", alerts[0])

    def test_redis_down_never_blocks_a_decision(self):
        with mock.patch("memory_store.redis_state.get_redis", return_value=None):
            self.assertFalse(cr._decision_route_exhausted("typesafe"))
            cr._mark_decision_route_exhausted("typesafe", "t", "quota", "x")  # no raise




class OwnerAlertQueueTests(unittest.TestCase):
    def test_alerts_queue_in_order_and_drain(self):
        class FakeRedis:
            def __init__(self):
                self.items = []
            def rpush(self, key, value):
                self.items.append(value)
            def ltrim(self, key, start, end):
                self.items = self.items[start:] if end == -1 else self.items[start:end + 1]
            def lpop(self, key):
                return self.items.pop(0) if self.items else None
        from memory_store import redis_state
        fake = FakeRedis()
        with mock.patch.object(redis_state, "get_redis", return_value=fake):
            for i in range(55):
                self.assertTrue(redis_state.push_owner_alert(f"a{i}"))
            self.assertEqual(len(fake.items), 50)  # oldest dropped past the cap
            self.assertEqual(redis_state.pop_owner_alerts(3), ["a5", "a6", "a7"])
        with mock.patch.object(redis_state, "get_redis", return_value=None):
            self.assertFalse(redis_state.push_owner_alert("x"))
            self.assertEqual(redis_state.pop_owner_alerts(), [])


if __name__ == "__main__":
    unittest.main()
