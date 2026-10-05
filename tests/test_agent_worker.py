"""Agent worker: request checks, MCP endpoint, sources, and the run loop with a fake model."""
import asyncio
import hashlib
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient  # noqa: E402

from worker import endpoint as api, runner, service  # noqa: E402

SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}, "sources": {"type": "array", "items": {"type": "string"}}},
          "required": ["answer"], "additionalProperties": False}


def request(**extra):
    return {"instructions": "Find the year.", "tools": ["fetch_url"], "resultSchema": SCHEMA, "tier": "review",
            "budgetUsd": 0.1, **extra}


class RequestChecks(unittest.TestCase):
    def test_valid_request_is_normalized(self):
        normalized = runner.validate_request(request())
        self.assertEqual(normalized["maxRounds"], 12)

    def test_rejections(self):
        cases = [
            (request(tools=["save_diary"]), "tools must be chosen"),
            (request(resultSchema={"type": "array"}), "type object"),
            (request(tier="opus"), "tier"),
            (request(budgetUsd=5), "budgetUsd"),
            (request(maxRounds=99), "maxRounds"),
            (request(validator={"tool": "editorial_store", "arguments": {}}), "validator"),
            (request(instructions=""), "instructions"),
            (request(sources=[{"url": "x"}]), "sources"),
        ]
        for value, message in cases:
            with self.assertRaisesRegex(ValueError, message):
                runner.validate_request(value)


class Endpoint(unittest.TestCase):
    TOKEN = "t" * 43

    def setUp(self):
        clients = {"frontend": hashlib.sha256(self.TOKEN.encode()).digest()}
        self.client = TestClient(api.create_app(clients))

    def rpc(self, method, params=None, token=TOKEN):
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        return self.client.post("/worker/mcp", json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}, headers=headers)

    def test_auth_and_listing(self):
        self.assertEqual(self.rpc("ping", token=None).status_code, 401)
        self.assertEqual(self.rpc("ping", token="x" * 43).status_code, 401)
        names = [t["name"] for t in self.rpc("tools/list").json()["result"]["tools"]]
        self.assertEqual(names, ["agent_task_submit", "agent_task_get", "agent_task_cancel"])
        self.assertEqual(self.rpc("initialize", {"protocolVersion": "2025-03-26"}).json()["result"]["protocolVersion"], "2025-03-26")

    def test_submit_validates_and_records_the_client(self):
        with patch.object(api.store, "submit", return_value={"taskId": "5", "status": "queued"}) as submit:
            reply = self.rpc("tools/call", {"name": "agent_task_submit", "arguments": {"idempotencyKey": "job:1", "request": request()}})
        self.assertEqual(reply.json()["result"]["structuredContent"], {"taskId": "5", "status": "queued"})
        self.assertEqual(submit.call_args.args[:2], ("frontend", "job:1"))
        bad = self.rpc("tools/call", {"name": "agent_task_submit", "arguments": {"idempotencyKey": "job:1", "request": request(tier="x")}})
        self.assertTrue(bad.json()["result"]["isError"])
        self.assertEqual(bad.json()["result"]["structuredContent"]["status"], 400)

    def test_get_scopes_to_the_client(self):
        with patch.object(api.store, "get", return_value=None) as get:
            reply = self.rpc("tools/call", {"name": "agent_task_get", "arguments": {"taskId": "9"}})
        self.assertEqual(get.call_args.args, ("frontend", "9"))
        self.assertEqual(reply.json()["result"]["structuredContent"]["status"], 404)
        self.assertTrue(self.rpc("tools/call", {"name": "agent_task_get", "arguments": {"taskId": "x"}}).json()["result"]["isError"])


class Executor(unittest.TestCase):
    def test_cancelled_rows_stop_their_running_task(self):
        async def scenario():
            job = asyncio.create_task(asyncio.sleep(30))
            running = {7: job}
            with patch.object(service.store, "is_cancelled", return_value=True), \
                    patch.object(service.store, "claim", return_value=None):
                await service.poll_once(running)
            await asyncio.sleep(0)
            return job.cancelled()
        self.assertTrue(asyncio.run(scenario()))


class SourcesRecord(unittest.TestCase):
    def test_ids_are_stable_per_page(self):
        sources = runner.Sources([{"url": "https://a", "text": "supplied"}])

        async def fetch(url):
            return f"page {url}"
        wrapped = sources.wrap("fetch_url", fetch)
        first = asyncio.run(wrapped(url="https://b"))
        again = asyncio.run(wrapped(url="https://b"))
        self.assertTrue(first.startswith("[Source ID: S2]"))
        self.assertTrue(again.startswith("[Source ID: S2]"))
        self.assertEqual([s["id"] for s in sources.items], ["S1", "S2"])
        self.assertEqual(sources.items[1]["sha256"], hashlib.sha256(b"page https://b").hexdigest())


class FakeBinding:
    """Plays a model: opens one page, submits an invalid then a valid result."""
    client, model, render_provider, reasoning = None, "fake", "deepseek", {}

    def __init__(self, submissions):
        self.submissions = submissions

    async def chat(self, messages, *, tool_handlers, budget_tracker, **_):
        from tool_gateway.results import ToolRejection
        await tool_handlers["fetch_url"](url="https://example.org")
        budget_tracker["total_cost"] = 0.012
        for value in self.submissions:
            try:
                await tool_handlers[runner.RESULT_TOOL](**value)
                return
            except ToolRejection:
                continue


def run_with(submissions, **extra):
    policy = SimpleNamespace(max_rounds=12, max_output_tokens=1000, max_input_tokens=1000, max_output_continuations=0)

    async def fetch_url(url):
        return "The congress met in 1903."
    with patch("bot_config.resolve_agent_tool_loop", return_value=FakeBinding(submissions)), \
            patch("tool_gateway.inference.resolve_agent_inference_policy", return_value=policy), \
            patch.dict("runtime_tools.registry.TOOL_HANDLERS", {"fetch_url": fetch_url}):
        return asyncio.run(runner.run(1, runner.validate_request(request(**extra))))


class RunLoop(unittest.TestCase):
    def test_schema_rejection_then_accepted_result(self):
        outcome = run_with([{"answer": 1903}, {"answer": "1903", "sources": ["S1"]}])
        self.assertEqual(outcome["result"], {"answer": "1903", "sources": ["S1"]})
        self.assertEqual(outcome["sources"][0]["arguments"], {"url": "https://example.org"})
        self.assertEqual(outcome["usage"]["costUsd"], 0.012)
        self.assertEqual(len(outcome["rejections"]), 1)

    def test_remote_validator_rejection_is_fed_back(self):
        from commulingo.mcp_client import CommuLingoToolError
        replies = [CommuLingoToolError({"error": "bio is too long", "status": 400}), {"status": "validated"}]

        def call_tool(name, arguments):
            self.assertEqual(name, "editorial_validate")
            self.assertEqual(arguments["id"], "lenin")
            reply = replies.pop(0)
            if isinstance(reply, Exception):
                raise reply
            return reply
        with patch("commulingo.mcp_client.call_tool", side_effect=call_tool):
            outcome = run_with([{"answer": "long"}, {"answer": "short"}],
                               validator={"tool": "editorial_validate", "arguments": {"id": "lenin"}})
        self.assertEqual(outcome["result"], {"answer": "short"})
        self.assertEqual(outcome["validation"], {"status": "validated"})
        self.assertEqual(outcome["rejections"], ["bio is too long"])

    def test_no_accepted_result_fails_with_report(self):
        with self.assertRaises(runner.TaskFailed) as caught:
            run_with([{"answer": 1}])
        self.assertIn("without an accepted result", str(caught.exception))
        self.assertEqual(caught.exception.report["usage"]["costUsd"], 0.012)


if __name__ == "__main__":
    unittest.main()
