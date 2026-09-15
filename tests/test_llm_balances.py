from __future__ import annotations

import asyncio
import json
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from llm_proxy.app import (
    BILLING_PROVIDERS,
    billing,
    billing_headers,
    billing_request,
    normalize_billing_response,
)
from scripts.llm_balances import (
    format_telegram_report,
    official_summary,
)


class TestBillingProxy(unittest.TestCase):
    def test_admin_credentials_only_have_fixed_billing_specs(self):
        self.assertEqual(BILLING_PROVIDERS["openai"]["secret"], "OPENAI_ADMIN_KEY")
        self.assertEqual(BILLING_PROVIDERS["claude"]["secret"], "ANTHROPIC_ADMIN_KEY")
        self.assertNotIn("openai_admin", __import__("llm_proxy.app", fromlist=["PROVIDERS"]).PROVIDERS)
        self.assertNotIn("anthropic_admin", __import__("llm_proxy.app", fromlist=["PROVIDERS"]).PROVIDERS)

    def test_cost_request_windows(self):
        now = datetime(2026, 8, 29, 12, 0, tzinfo=timezone.utc)
        _, openai = billing_request("openai", 30, now=now)
        self.assertEqual(openai["end_time"] - openai["start_time"], 30 * 86400)
        self.assertEqual(openai["limit"], 31)
        _, claude = billing_request("claude", 30, now=now)
        self.assertEqual(claude["starting_at"], "2026-07-30T12:00:00Z")
        self.assertEqual(claude["ending_at"], "2026-08-29T12:00:00Z")

    def test_admin_header_styles(self):
        self.assertEqual(billing_headers("bearer", "K"), {
            "authorization": "Bearer K", "accept": "application/json",
        })
        anthropic = billing_headers("anthropic-admin", "K")
        self.assertEqual(anthropic["x-api-key"], "K")
        self.assertEqual(anthropic["anthropic-version"], "2023-06-01")

    def test_balance_normalization(self):
        deepseek = normalize_billing_response("deepseek", {
            "is_available": True,
            "balance_infos": [{
                "currency": "USD", "total_balance": "12.28",
                "granted_balance": "0.00", "topped_up_balance": "12.28",
            }],
        }, 30)
        self.assertEqual(deepseek["balances"][0]["available"], 12.28)
        self.assertTrue(deepseek["can_call"])

        kimi = normalize_billing_response("kimi", {
            "status": True,
            "data": {
                "available_balance": 14.45891,
                "voucher_balance": 4.45891,
                "cash_balance": 10,
            },
        }, 30)
        self.assertEqual(kimi["balances"][0]["granted"], 4.45891)
        self.assertEqual(kimi["balances"][0]["cash"], 10.0)

    def test_cost_normalization(self):
        openai = normalize_billing_response("openai", {
            "data": [{"results": [
                {"amount": {"value": 1.25, "currency": "usd"}},
                {"amount": {"value": 2, "currency": "usd"}},
            ]}],
            "has_more": False,
        }, 30)
        self.assertEqual(openai["costs"], [{"currency": "USD", "amount": 3.25}])

        claude = normalize_billing_response("claude", {
            "data": [{"results": [
                {"amount": "75.25", "currency": "USD"},
                {"amount": "24.75", "currency": "USD"},
            ]}],
        }, 30)
        self.assertEqual(claude["costs"], [{"currency": "USD", "amount": 1.0}])

    def test_missing_admin_key_is_structured_fallback(self):
        with patch("llm_proxy.app._credential", return_value=""):
            response = asyncio.run(billing("openai", 30))
        self.assertEqual(response.status_code, 200)
        payload = json.loads(response.body)
        self.assertEqual(payload["status"], "credential_missing")
        self.assertEqual(payload["required_credential"], "OPENAI_ADMIN_KEY")


class TestBalanceCli(unittest.TestCase):
    def test_official_cost_providers_never_display_estimates(self):
        for provider in ("claude", "openai"):
            for status in ("ok", "credential_missing", "upstream_error"):
                report = {"window_days": 30, "providers": [{
                    "provider": provider,
                    "official": {"status": status, "kind": "cost",
                                 "costs": [{"currency": "USD", "amount": 30.2184}]},
                    "local_audit": {"calls": 11, "spend_usd": 495.0},
                }]}
                rendered = format_telegram_report(report)
                self.assertNotIn("추정", rendered)
                self.assertNotIn("495", rendered)

    def test_summary_distinguishes_balance_cost_and_fallback(self):
        self.assertIn("공식 잔액: USD 12.2800", official_summary({
            "status": "ok", "kind": "balance",
            "balances": [{"currency": "USD", "available": 12.28}],
        }, 30))
        self.assertIn("공식 사용 비용 (30일): USD 3.2500", official_summary({
            "status": "ok", "kind": "cost",
            "costs": [{"currency": "USD", "amount": 3.25}],
        }, 30))
        self.assertIn("OPENAI_ADMIN_KEY", official_summary({
            "status": "credential_missing", "required_credential": "OPENAI_ADMIN_KEY",
        }, 30))

    def test_telegram_report_keeps_deepseek_estimate(self):
        report = {
            "window_days": 7,
            "local_audit_error": None,
            "providers": [{
                "provider": "deepseek",
                "official": {
                    "status": "ok", "kind": "balance",
                    "balances": [{"currency": "USD", "available": 12.28}],
                },
                "local_audit": {"calls": 42, "spend_usd": 1.2345},
            }],
        }
        rendered = format_telegram_report(report)
        self.assertIn("최근 7일", rendered)
        self.assertIn("공식 잔액: USD 12.2800", rendered)
        self.assertIn("사용 비용 추정", rendered)
        self.assertIn("1.2345", rendered)


if __name__ == "__main__":
    unittest.main()
