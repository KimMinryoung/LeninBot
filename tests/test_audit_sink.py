"""Hermetic tests for audit_sink.py and the proxy's /audit endpoints.

No DB, no network: inserts and HTTP are patched. Covers the column
whitelist, mode resolution, the client POST path (retry on connection
error, no retry on 4xx), the gateway workers' batching through the sink,
and the FastAPI endpoints with insert_rows stubbed.
"""

import json
import os
import sys
import unittest
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ["LENINBOT_LLM_AUDIT_DB"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import audit_sink  # noqa: E402


class NormalizeTests(unittest.TestCase):
    def test_coerces_and_caps(self):
        row = audit_sink.normalize_row("llm", {
            "surface": "loop", "status": "ok", "tokens_in": "12", "cost_usd": "0.5",
            "error_excerpt": "x" * 2000, "label": None,
        })
        self.assertEqual(row["tokens_in"], 12)
        self.assertEqual(row["cost_usd"], 0.5)
        self.assertEqual(len(row["error_excerpt"]), 1001)  # 1000 + ellipsis
        self.assertIsNone(row["label"])
        self.assertEqual(set(row), set(audit_sink.TABLES["llm"]["columns"]))

    def test_rejects_unknown_column_and_missing_required(self):
        with self.assertRaises(ValueError):
            audit_sink.normalize_row("llm", {"surface": "loop", "status": "ok", "bogus": 1})
        with self.assertRaises(ValueError):
            audit_sink.normalize_row("tool", {"tool_name": "x"})  # decision missing
        with self.assertRaises(ValueError):
            audit_sink.normalize_row("nope", {})
        with self.assertRaises(ValueError):
            audit_sink.normalize_rows("llm", [{}] * (audit_sink.MAX_ROWS_PER_REQUEST + 1))

    def test_tool_row_bools_and_json_strings(self):
        row = audit_sink.normalize_row("tool", {
            "tool_name": "web_search", "decision": "allow", "is_owner": 1,
            "args_summary": {"q": "x"}, "chat_log_id": "7",
        })
        self.assertIs(row["is_owner"], True)
        self.assertEqual(row["args_summary"], '{"q": "x"}')
        self.assertEqual(row["chat_log_id"], 7)

    def test_insert_sql_lists_every_column(self):
        for kind, spec in audit_sink.TABLES.items():
            sql = audit_sink._INSERT_SQL[kind]
            self.assertTrue(sql.startswith(f"INSERT INTO {spec['table']} ("))
            for col in spec["columns"]:
                self.assertIn(f"%({col})s", sql)


class ModeTests(unittest.TestCase):
    def setUp(self):
        self._prev_local = audit_sink._local_sink  # another module may have imported the proxy app
        audit_sink.set_local_sink(False)

    def tearDown(self):
        audit_sink.set_local_sink(self._prev_local)
        os.environ.pop("LENINBOT_AUDIT_SINK", None)

    def test_local_wins(self):
        audit_sink.set_local_sink(True)
        with patch.object(audit_sink, "proxy_base", return_value="http://127.0.0.1:8110"):
            self.assertEqual(audit_sink.mode(), "local")

    def test_proxy_when_configured_else_db(self):
        with patch.object(audit_sink, "proxy_base", return_value="http://127.0.0.1:8110"):
            self.assertEqual(audit_sink.mode(), "proxy")
        with patch.object(audit_sink, "proxy_base", return_value=None):
            self.assertEqual(audit_sink.mode(), "db")

    def test_env_override(self):
        os.environ["LENINBOT_AUDIT_SINK"] = "db"
        with patch.object(audit_sink, "proxy_base", return_value="http://127.0.0.1:8110"):
            self.assertEqual(audit_sink.mode(), "db")


class _Resp:
    def __init__(self, status=200, payload=None):
        self.status = status
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, n=-1):
        return json.dumps(self._payload or {}).encode()


class PostTests(unittest.TestCase):
    def setUp(self):
        audit_sink._post_failed_before = False

    def test_posts_json_to_proxy(self):
        seen = {}

        def fake_urlopen(req, timeout=None):
            seen["url"] = req.full_url
            seen["body"] = json.loads(req.data)
            return _Resp(200)

        with patch.object(audit_sink, "proxy_base", return_value="http://p"), \
             patch.object(audit_sink.urllib.request, "urlopen", fake_urlopen):
            ok = audit_sink.post_rows("tool", [{"tool_name": "t", "decision": "allow"}])
        self.assertTrue(ok)
        self.assertEqual(seen["url"], "http://p/audit/tool")
        self.assertEqual(seen["body"][0]["tool_name"], "t")

    def test_retries_connection_error_once(self):
        calls = []

        def flaky(req, timeout=None):
            calls.append(1)
            if len(calls) == 1:
                raise urllib.error.URLError("connection refused")
            return _Resp(200)

        with patch.object(audit_sink, "proxy_base", return_value="http://p"), \
             patch.object(audit_sink.urllib.request, "urlopen", flaky), \
             patch.object(audit_sink.time, "sleep"):
            self.assertTrue(audit_sink.post_rows("llm", [{"surface": "loop", "status": "ok"}]))
        self.assertEqual(len(calls), 2)

    def test_no_retry_on_4xx(self):
        calls = []

        def bad(req, timeout=None):
            calls.append(1)
            raise urllib.error.HTTPError(req.full_url, 400, "bad", {}, None)

        with patch.object(audit_sink, "proxy_base", return_value="http://p"), \
             patch.object(audit_sink.urllib.request, "urlopen", bad):
            self.assertFalse(audit_sink.post_rows("llm", [{"surface": "loop", "status": "ok"}]))
        self.assertEqual(len(calls), 1)

    def test_no_proxy_drops(self):
        with patch.object(audit_sink, "proxy_base", return_value=None):
            self.assertFalse(audit_sink.post_rows("llm", [{"surface": "loop", "status": "ok"}]))

    def test_fetch_today_spend(self):
        with patch.object(audit_sink, "proxy_base", return_value="http://p"), \
             patch.object(audit_sink.urllib.request, "urlopen",
                          lambda url, timeout=None: _Resp(200, {"spend": {"claude": "1.5"}})):
            self.assertEqual(audit_sink.fetch_today_spend(), {"claude": 1.5})


class GatewayWorkerTests(unittest.TestCase):
    """Both gateways drain their queues through audit_sink."""

    def test_llm_gateway_batches_to_proxy(self):
        import llm.gateway as gw
        posted = []
        with patch.object(audit_sink, "mode", return_value="proxy"), \
             patch.object(audit_sink, "post_rows", lambda kind, rows: posted.append((kind, rows)) or True):
            gw._drain_batch([{"surface": "loop", "status": "ok"}, {"surface": "loop", "status": "error"}])
        self.assertEqual(posted[0][0], "llm")
        self.assertEqual(len(posted[0][1]), 2)

    def test_llm_gateway_direct_when_local(self):
        import llm.gateway as gw
        inserted = []
        with patch.object(audit_sink, "mode", return_value="local"), \
             patch.object(audit_sink, "insert_rows", lambda kind, rows: inserted.append((kind, len(rows))) or len(rows)):
            gw._drain_batch([{"surface": "proxy", "status": "ok"}])
        self.assertEqual(inserted, [("llm", 1)])

    def test_llm_gateway_direct_failure_is_swallowed(self):
        import llm.gateway as gw

        def boom(kind, rows):
            raise RuntimeError("no db")

        with patch.object(audit_sink, "mode", return_value="db"), \
             patch.object(audit_sink, "insert_rows", boom):
            gw._drain_batch([{"surface": "loop", "status": "ok"}])  # must not raise

    def test_tool_audit_batches_to_proxy(self):
        import importlib
        ta = importlib.import_module("security_gateway.audit")  # package exports audit() under the same name
        posted = []
        with patch.object(audit_sink, "mode", return_value="proxy"), \
             patch.object(audit_sink, "post_rows", lambda kind, rows: posted.append(kind) or True):
            ta._drain_batch([{"tool_name": "t", "decision": "allow"}])
        self.assertEqual(posted, ["tool"])

    def test_spend_via_proxy_or_db(self):
        import llm.gateway as gw
        gw._spend_cache = None
        with patch.object(audit_sink, "mode", return_value="proxy"), \
             patch.object(audit_sink, "fetch_today_spend", return_value={"claude": 2.0}):
            self.assertEqual(gw._today_spend(), {"claude": 2.0})
        gw._spend_cache = None
        with patch.object(audit_sink, "mode", return_value="local"), \
             patch.object(audit_sink, "today_spend", return_value={"openai": 1.0}):
            self.assertEqual(gw._today_spend(), {"openai": 1.0})
        gw._spend_cache = None

    def test_adhoc_rows_are_labelled(self):
        import llm.gateway as gw
        captured = []
        with patch.dict(os.environ, {"LENINBOT_LLM_AUDIT_DB": "1"}), \
             patch.dict(os.environ, {"INVOCATION_ID": "", "LENINBOT_SERVICE": ""}), \
             patch.object(gw, "_ensure_worker"), \
             patch.object(gw._DB_QUEUE, "put_nowait", captured.append):
            gw.record_llm_call(surface="oneshot", caller="c", model="gemini-embedding-001", label="embed")
        self.assertEqual(captured[0]["label"], "embed [adhoc]")


class ProxyEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import importlib
        from fastapi.testclient import TestClient
        import llm_proxy.app as proxy_app
        cls._prev_local = audit_sink._local_sink
        audit_sink.set_local_sink(False)
        proxy_app = importlib.reload(proxy_app)  # re-run the import-time set_local_sink(True)
        cls.app_module = proxy_app
        cls.client = TestClient(proxy_app.app)

    @classmethod
    def tearDownClass(cls):
        audit_sink.set_local_sink(cls._prev_local)

    def test_import_marks_process_as_local_sink(self):
        self.assertEqual(audit_sink.mode(), "local")

    def test_ingest_ok(self):
        seen = {}
        with patch.object(audit_sink, "insert_rows", lambda kind, rows: seen.setdefault(kind, rows) and len(rows)):
            r = self.client.post("/audit/tool", json=[{"tool_name": "t", "decision": "allow", "is_owner": True}])
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json(), {"inserted": 1})
        self.assertEqual(seen["tool"][0]["tool_name"], "t")

    def test_ingest_rejects_bad_payload(self):
        r = self.client.post("/audit/llm", json=[{"surface": "loop", "status": "ok", "bogus": 1}])
        self.assertEqual(r.status_code, 400)
        r = self.client.post("/audit/nope", json=[])
        self.assertEqual(r.status_code, 404)
        r = self.client.post("/audit/llm", content=b"not json", headers={"content-type": "application/json"})
        self.assertEqual(r.status_code, 400)

    def test_ingest_db_down_is_503(self):
        def boom(kind, rows):
            raise RuntimeError("db down")

        with patch.object(audit_sink, "insert_rows", boom):
            r = self.client.post("/audit/llm", json=[{"surface": "loop", "status": "ok"}])
        self.assertEqual(r.status_code, 503)

    def test_payload_cap(self):
        big = b"[" + b"1" * (audit_sink.MAX_BODY_BYTES + 10) + b"]"
        r = self.client.post("/audit/llm", content=big, headers={"content-type": "application/json"})
        self.assertEqual(r.status_code, 413)

    def test_spend_endpoint(self):
        with patch.object(audit_sink, "today_spend", return_value={"claude": 3.25}):
            r = self.client.get("/audit/spend/today")
        self.assertEqual(r.json(), {"spend": {"claude": 3.25}})

    def test_health_reports_sink_without_changing_status(self):
        with patch.object(self.app_module, "_credential", return_value="k"), \
             patch.object(audit_sink, "sink_health", return_value="error: OperationalError"):
            r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["audit_sink"], "error: OperationalError")


if __name__ == "__main__":
    unittest.main()
