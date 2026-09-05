import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from scripts.commulingo_research_memory import ResearchMemory, RETENTION
from tool_gateway.observations import argument_rejection_observer
from tool_gateway.results import ToolFailure, ToolRejection


class ResearchMemoryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "memory.sqlite3"
        self.memory = ResearchMemory("gap:974", path=self.path)

    async def test_restart_reuses_complete_evidence_and_isolates_targets(self):
        provider = AsyncMock(return_value='<external source="url:https://example.com">Evidence</external>')
        await self.memory.wrap({"web_search": provider})["web_search"](query="exact question")
        restarted = ResearchMemory("gap:974", path=self.path)
        result = await restarted.wrap({"web_search": provider})["web_search"](query="exact question")
        self.assertIn("Evidence", result)
        self.assertIn("exact question", restarted.context())
        self.assertIn("retrieved_at_unix", restarted.context())
        self.assertEqual(provider.await_count, 1)
        other = ResearchMemory("gap:712", path=self.path)
        self.assertNotIn("Evidence", other.context())
        await other.wrap({"web_search": provider})["web_search"](query="exact question")
        self.assertEqual(provider.await_count, 2)

    async def test_failed_reads_are_not_reused(self):
        provider = AsyncMock(side_effect=[ToolFailure("unavailable"), "usable"])
        handler = self.memory.wrap({"web_search": provider})["web_search"]
        self.assertIsInstance(await handler(query="q"), ToolFailure)
        self.assertEqual(await handler(query="q"), "usable")
        self.assertEqual(provider.await_count, 2)

    async def test_recency_expiry_and_explicit_refresh(self):
        provider = AsyncMock(return_value="evidence")
        handler = self.memory.wrap({"web_search": provider})["web_search"]
        with patch("scripts.commulingo_research_memory.time.time", return_value=1000):
            await handler(query="q", topic="news")
        with patch("scripts.commulingo_research_memory.time.time", return_value=1061):
            self.assertNotIn('"tool":"web_search"', self.memory.context())
            await handler(query="q", topic="news")
            await handler(query="q", topic="news", use_cache=False)
        self.assertEqual(provider.await_count, 3)

    async def test_format_rejection_retains_draft_and_repair_avoids_research(self):
        provider = AsyncMock(return_value="source text")
        writer = AsyncMock(side_effect=[ToolRejection("validation_failed: 901 characters exceeds limit"), "saved"])
        handlers = self.memory.wrap({"fetch_url": provider, "commulingo_person_create": writer})
        await handlers["fetch_url"](url="https://example.com")
        draft = {"person_id": "kontorovich", "fields": {"bio": "long draft"}, "citations": [{"url": "https://example.com"}]}
        with self.assertRaises(ToolRejection):
            await handlers["commulingo_person_create"](**draft)
        restarted = ResearchMemory("gap:974", path=self.path)
        self.assertIn("long draft", restarted.context())
        self.assertIn("901 characters", restarted.context())
        handlers = restarted.wrap({"fetch_url": provider, "commulingo_person_create": writer})
        self.assertEqual(await handlers["fetch_url"](url="https://example.com"), "source text")
        with self.assertRaises(ToolRejection):
            await handlers["fetch_url"](url="https://other.com")
        self.assertEqual(provider.await_count, 1)
        self.assertEqual(await handlers["commulingo_person_create"](**draft), "saved")
        self.assertIsNone(restarted._draft())

    async def test_reference_lookup_remains_live_during_repair(self):
        self.memory.rejected("commulingo_person_create", {"person_id": "x"}, "unknown group")
        lookup = AsyncMock(return_value="current groups")
        handlers = self.memory.wrap({"commulingo_people": lookup})
        await handlers["commulingo_people"](action="list_groups")
        await handlers["commulingo_people"](action="list_groups")
        self.assertEqual(lookup.await_count, 2)

    async def test_evidence_rejection_still_allows_research(self):
        writer = AsyncMock(side_effect=ToolRejection("insufficient_evidence: unsupported claim"))
        provider = AsyncMock(return_value="additional source")
        handlers = self.memory.wrap({"web_search": provider, "commulingo_person_create": writer})
        with self.assertRaises(ToolRejection):
            await handlers["commulingo_person_create"](person_id="x")
        self.assertEqual(await handlers["web_search"](query="missing fact"), "additional source")

    async def test_gateway_schema_rejection_preserves_payload_before_handler(self):
        from tool_gateway.dispatcher import execute_tool

        writer = AsyncMock()
        decision = SimpleNamespace(denied=False, risk_class="write")
        with patch("tool_gateway.security.get_caller", return_value=SimpleNamespace()), \
             patch("tool_gateway.security.authorize", return_value=decision), \
             patch("tool_gateway.security.audit"):
            async def chat(messages, **kwargs):
                return await execute_tool(
                    "commulingo_person_create", {"fields": {"bio": "keep this draft"}},
                    kwargs["tool_handlers"], tool_schema={"type": "object", "properties": {
                        "person_id": {"type": "string"}, "fields": {"type": "object"}}, "required": ["person_id"]},
                )
            result, failed = await self.memory.chat(chat, [], tool_handlers={"commulingo_person_create": writer})
        self.assertTrue(failed)
        writer.assert_not_awaited()
        self.assertIn("keep this draft", self.memory.context())
        self.assertIn("required", self.memory.context())
        self.assertIsNone(argument_rejection_observer.get())

    async def test_attempt_output_survives_next_conversation(self):
        first = AsyncMock(return_value="Unfinished researched draft")
        await self.memory.chat(first, [{"role": "user", "content": "task"}], tool_handlers={})
        restarted = ResearchMemory("gap:974", path=self.path)
        second = AsyncMock(return_value="done")
        await restarted.chat(second, [{"role": "user", "content": "task"}], tool_handlers={})
        self.assertIn("Unfinished researched draft", second.call_args.args[0][-1]["content"])
        self.assertIsNone(argument_rejection_observer.get())

    async def test_observer_resets_after_exception(self):
        with self.assertRaises(RuntimeError):
            await self.memory.chat(AsyncMock(side_effect=RuntimeError("stopped")), [], tool_handlers={})
        self.assertIsNone(argument_rejection_observer.get())

    async def test_real_stage_retry_keeps_evidence_and_rejected_draft(self):
        from scripts import commulingo_people_maintainer as maintainer

        state = {"attempt": 0, "writes": 0}
        research = AsyncMock(return_value="verified source body")

        async def write(**args):
            if args["fields"]["bio"] == "too long":
                raise ToolRejection("validation_failed: too many characters")
            state["writes"] += 1
            return "saved"

        async def chat(messages, **kwargs):
            state["attempt"] += 1
            handlers = kwargs["tool_handlers"]
            if state["attempt"] == 1:
                await handlers["fetch_url"](url="https://example.com/source")
                try:
                    await handlers["commulingo_person_create"](
                        person_id="test", fields={"bio": "too long"}, citations=["https://example.com/source"])
                except ToolRejection:
                    pass
                return "unfinished draft"
            context = messages[-1]["content"]
            self.assertIn("verified source body", context)
            self.assertIn("too long", context)
            self.assertIn("https://example.com/source", context)
            self.assertIn("unfinished draft", context)
            self.assertEqual(await handlers["fetch_url"](url="https://example.com/source"), "verified source body")
            return await handlers["commulingo_person_create"](
                person_id="test", fields={"bio": "short"}, citations=["https://example.com/source"])

        binding = SimpleNamespace(chat=chat, client=None, model="test", render_provider="test", reasoning={})
        policy = SimpleNamespace(max_output_continuations=1, max_rounds=16, max_output_tokens=4096,
                                 max_input_tokens=160000, budget_usd=0.35)
        spec = SimpleNamespace(name="commulingo_curator", render_prompt=lambda **kwargs: "system")
        with patch.object(maintainer, "resolve_agent_tool_loop", return_value=binding), \
             patch.object(maintainer, "completed_run_count", side_effect=lambda: state["writes"]), \
             patch("scripts.commulingo_research_memory.STORE_PATH", self.path):
            result, _, _ = await maintainer._call_curator_stage(
                task="repair person", spec=spec, tools=[],
                handlers={"fetch_url": research, "commulingo_person_create": write},
                policy=policy, stage="test", expect_edit=True, before_count=0,
                finalization_tools=["commulingo_person_create"], terminal_tools=["commulingo_person_create"],
            )
        self.assertEqual(result, "saved")
        self.assertEqual(state["attempt"], 2)
        self.assertEqual(research.await_count, 1)

    async def test_retention_expires_persisted_evidence(self):
        provider = AsyncMock(return_value="old evidence")
        with patch("scripts.commulingo_research_memory.time.time", return_value=1000):
            await self.memory.wrap({"web_search": provider})["web_search"](query="q")
        with patch("scripts.commulingo_research_memory.time.time", return_value=1001 + RETENTION):
            restarted = ResearchMemory("gap:974", path=self.path)
            self.assertNotIn("old evidence", restarted.context())


if __name__ == "__main__":
    unittest.main()
