"""CommuLingo admin MCP client: request mapping and error shapes against a fake server."""
import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from commulingo import mcp_client  # noqa: E402
from commulingo.person_service import apply_person_spec, call_person_service  # noqa: E402
from commulingo.pipeline import service as pipeline_service  # noqa: E402

CALLS = []
REPLIES = {}


class FakeMcp(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        params = body["params"]
        CALLS.append({"auth": self.headers["Authorization"], "name": params["name"], "arguments": params["arguments"]})
        reply = REPLIES.get(params["name"], {"structuredContent": {"ok": True}})
        if callable(reply):
            reply = reply(params["arguments"])
        payload = json.dumps({"jsonrpc": "2.0", "id": body["id"], "result": reply}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


def urllib_response(message):
    import io

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False
    return Response(json.dumps(message).encode())


def error(payload):
    return {"isError": True, "structuredContent": payload}


class CommuLingoMcpClient(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), FakeMcp)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.env = patch.dict(os.environ, {"COMMULINGO_MCP_URL": f"http://127.0.0.1:{cls.server.server_port}/mcp",
                                          "COMMULINGO_MCP_TOKEN": "t" * 43})
        cls.env.start()

    @classmethod
    def tearDownClass(cls):
        cls.env.stop()
        cls.server.shutdown()

    def setUp(self):
        CALLS.clear()
        REPLIES.clear()

    def test_reads_map_to_get_tools_and_missing_is_none(self):
        REPLIES["person_get"] = {"structuredContent": {"person": {"id": "lenin", "revision": "v1-a"}}}
        REPLIES["term_get"] = error({"error": "term not found: x", "status": 404})
        self.assertEqual(call_person_service({"command": "read", "id": "lenin"})["revision"], "v1-a")
        self.assertIsNone(call_person_service({"command": "read", "target": "term", "id": "x"}))
        self.assertEqual(pipeline_service.call({"command": "read", "target": "person_section", "id": "lenin"})["id"], "lenin")
        self.assertEqual([c["name"] for c in CALLS], ["person_get", "term_get", "person_get"])
        self.assertEqual(CALLS[0]["auth"], "Bearer " + "t" * 43)

    def test_store_and_pipeline_requests_keep_their_payload(self):
        call_person_service({"command": "submit", "target": "term", "action": "update", "id": "nep",
                             "fields": {"expectedRevision": "v1-b"}, "sources": ["s"], "changedBy": "commulingo-maintainer"})
        call_person_service({"command": "review", "suggestionId": "7", "approve": True, "note": "ok"})
        pipeline_service.call({"command": "validate", "target": "person", "id": "lenin", "fields": {}, "changedBy": "commulingo-pipeline"})
        self.assertEqual(CALLS[0], {"auth": CALLS[0]["auth"], "name": "editorial_store", "arguments": {
            "command": "submit", "target": "term", "changedBy": "commulingo-maintainer",
            "request": {"action": "update", "id": "nep", "fields": {"expectedRevision": "v1-b"}, "sources": ["s"]}}})
        self.assertEqual(CALLS[1]["arguments"], {"command": "review", "target": "person",
                                                 "request": {"suggestionId": "7", "approve": True, "note": "ok"}})
        self.assertEqual(CALLS[2]["arguments"], {"command": "validate", "target": "person",
                                                 "request": {"id": "lenin", "fields": {}, "changedBy": "commulingo-pipeline"}})

    def test_tool_errors_keep_the_old_rpc_message_shape(self):
        REPLIES["editorial_store"] = error({"error": "stale", "status": 409, "code": "revision_conflict", "currentRevision": "v1-c"})
        with self.assertRaises(ValueError) as caught:
            call_person_service({"command": "submit", "id": "lenin", "fields": {}})
        self.assertIn("revision_conflict", str(caught.exception))
        self.assertEqual(caught.exception.current_revision, "v1-c")

    def test_people_spec_goes_through_upsert(self):
        REPLIES["people_upsert"] = {"structuredContent": {"dryRun": False, "results": [
            {"id": "lenin", "status": "approved", "suggestionId": "9", "sections": [{"slug": "exile", "status": "approved"}]}]}}
        with tempfile.NamedTemporaryFile("w", suffix=".json") as spec:
            json.dump({"people": [{"id": "lenin"}], "changedBy": "backfill"}, spec)
            spec.flush()
            output = apply_person_spec(spec.name)
        self.assertIn("committed 1 person", output)
        self.assertIn("  approved section exile", output)
        self.assertEqual(CALLS[0]["arguments"], {"people": [{"id": "lenin"}], "dryRun": False, "changedBy": "backfill"})
        REPLIES["people_upsert"] = error({"error": "bad card", "status": 400})
        with tempfile.NamedTemporaryFile("w", suffix=".json") as spec:
            json.dump({"people": [{"id": "lenin"}]}, spec)
            spec.flush()
            with self.assertRaisesRegex(RuntimeError, "bad card"):
                apply_person_spec(spec.name)

    def test_reset_is_retried_for_reads_only(self):
        import http.client
        attempts = []

        def flaky(request, timeout):
            attempts.append(1)
            if len(attempts) == 1:
                raise http.client.RemoteDisconnected("starting")
            return urllib_response({"jsonrpc": "2.0", "id": 1, "result": {"structuredContent": {"person": {"id": "x"}}}})
        with patch.object(mcp_client, "REFUSED_RETRIES", (0,)), patch("urllib.request.urlopen", flaky):
            self.assertEqual(call_person_service({"command": "read", "id": "x"}), {"id": "x"})
        attempts.clear()
        with patch.object(mcp_client, "REFUSED_RETRIES", (0,)), patch("urllib.request.urlopen", flaky):
            with self.assertRaises(RuntimeError):
                call_person_service({"command": "submit", "id": "x", "fields": {}})
        self.assertEqual(len(attempts), 1, "a non-idempotent write is not resent")

    def test_unreachable_server_is_a_runtime_error(self):
        with patch.dict(os.environ, {"COMMULINGO_MCP_URL": "http://127.0.0.1:9/mcp"}), \
                patch.object(mcp_client, "REFUSED_RETRIES", (0,)):
            with self.assertRaises(RuntimeError):
                call_person_service({"command": "read", "id": "lenin"})


if __name__ == "__main__":
    unittest.main()
