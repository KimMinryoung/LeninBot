"""Async handlers keep DB work off the event loop and enforce their access checks."""
import asyncio
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient


class ChatHistoryAccessTests(unittest.TestCase):
    def setUp(self):
        from api_routes import chat_history
        self.module = chat_history
        app = FastAPI()
        app.include_router(chat_history.router)
        self.client = TestClient(app)

    def test_history_and_sessions_need_the_frontend_proxy(self):
        with patch.object(self.module, "trusted_proxy_request", return_value=False), \
                patch.object(self.module, "db_query") as query:
            for path in ("/history?fingerprint=abc", "/sessions?fingerprint=abc"):
                with self.subTest(path=path):
                    self.assertEqual(self.client.get(path).status_code, 403)
        query.assert_not_called()

    def test_proxied_history_queries_on_a_worker_thread(self):
        threads = []

        def query(*args, **kwargs):
            try:
                asyncio.get_running_loop()
                threads.append("event loop")
            except RuntimeError:
                threads.append("worker")
            return []

        with patch.object(self.module, "trusted_proxy_request", return_value=True), \
                patch.object(self.module, "db_query", side_effect=query):
            for path in ("/history?fingerprint=abc", "/sessions?fingerprint=abc"):
                self.assertEqual(self.client.get(path).status_code, 200)
        self.assertEqual(threads, ["worker", "worker"])


class X402ChallengeTests(unittest.TestCase):
    def test_bare_challenge_writes_no_ledger_row(self):
        from api_routes import x402_demo
        app = FastAPI()
        app.include_router(x402_demo.router)
        with patch("crypto_wallet.wallet.get_addresses", return_value={"base": "0x" + "1" * 40}), \
                patch("crypto_wallet.x402_ledger.record_x402_attempt") as record:
            response = TestClient(app).get("/x402-demo/quote")
        self.assertEqual(response.status_code, 402)
        self.assertIn("PAYMENT-REQUIRED", response.headers)
        record.assert_not_called()


class AsyncPaidRequestTests(unittest.IsolatedAsyncioTestCase):
    async def test_reserve_and_settle_run_off_the_loop(self):
        from web_gateway import budget
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "policy.json"
            config.write_text(json.dumps({"daily_budget_usd": 1, "tavily_credit_usd": 0.008, "brave_search_usd": 0.005}))
            loop_thread = threading.current_thread()
            seen = []
            real_reserve = budget.reserve

            def reserve(*args):
                seen.append(threading.current_thread() is loop_thread)
                return real_reserve(*args)

            with patch.object(budget, "STORE_PATH", Path(tmp) / "usage.sqlite3"), \
                    patch.object(budget, "CONFIG_PATH", config), \
                    patch.object(budget, "reserve", side_effect=reserve):
                async with budget.paid_request_async("tavily", "search", "basic") as charge:
                    await charge.complete_async(1)
            self.assertEqual(seen, [False])
            self.assertTrue(charge.done)


class SpendLookupTests(unittest.IsolatedAsyncioTestCase):
    async def test_event_loop_never_waits_for_the_spend_lookup(self):
        import llm.gateway as gw
        release = threading.Event()
        calls = []

        def slow_spend():
            calls.append(1)
            release.wait(5)
            return {"claude": 3.0}

        gw._spend_cache = None
        self.addCleanup(setattr, gw, "_spend_cache", None)
        with patch("ops.audit_sink.mode", return_value="db"), \
                patch("ops.audit_sink.today_spend", side_effect=slow_spend):
            self.assertIsNone(gw._today_spend())   # cold cache: fail open, refresh behind
            self.assertIsNone(gw._today_spend())   # one refresh at a time
            release.set()
            for _ in range(100):
                if gw._spend_cache is not None:
                    break
                await asyncio.sleep(0.01)
            self.assertEqual(gw._today_spend(), {"claude": 3.0})
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
