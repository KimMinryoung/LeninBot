"""In-batch identity: one name, one node — even before the batch is saved."""
import unittest
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from graph_memory import structured_writer as writer
from kg_runtime import identity


class _Client:
    @asynccontextmanager
    async def session(self, **kwargs):
        yield object()


def _fact(subject, obj="1927년 상하이 쿠데타", **extra):
    return {"subject_name": subject, "subject_type": "Person", "predicate": "Involvement",
            "object_name": obj, "object_type": "Incident", "fact": f"{subject} took part", **extra}


class BatchIdentityTests(unittest.IsolatedAsyncioTestCase):
    async def _write(self, facts):
        graph = SimpleNamespace(driver=SimpleNamespace(client=_Client()), embedder=None)
        with patch.object(writer, "find_canonical_entity_uuid", new=AsyncMock(return_value=None)), \
             patch.object(writer, "_untrusted_aliases", new=AsyncMock(return_value=[])), \
             patch.object(writer, "_embed_in_batches", new=AsyncMock()), \
             patch.object(writer, "add_nodes_and_edges_bulk", new=AsyncMock()) as save, \
             patch.object(writer, "validate_episode_result", return_value=None):
            result = await writer.write_structured_facts(graph, facts, group_id="documents")
        return result, save.await_args.args[3]

    async def test_llm_side_reuses_source_node_created_in_same_batch(self):
        # LLM fact first, source-linked fact second: the order of the 2026-11-04 예쯔 twin.
        result, nodes = await self._write([
            _fact("예쯔"),
            _fact("예쯔", subject_external_id="commulingo:person:ye-zi", subject_aliases=["Ye Zi"]),
        ])
        self.assertEqual(result["status"], "ok")
        self.assertEqual([n.name for n in nodes].count("예쯔"), 1)

    async def test_llm_side_reuses_source_node_through_its_alias(self):
        _, nodes = await self._write([
            _fact("Ye Zi"),
            _fact("예쯔", subject_external_id="commulingo:person:ye-zi", subject_aliases=["Ye Zi"]),
        ])
        self.assertEqual(sorted(n.name for n in nodes if "Person" in n.labels), ["예쯔"])

    async def test_two_source_namesakes_leave_llm_side_unresolved(self):
        _, nodes = await self._write([
            _fact("블라디미르 코마로프", subject_external_id="commulingo:person:komarov-a"),
            _fact("블라디미르 코마로프", subject_external_id="commulingo:person:komarov-b"),
            _fact("블라디미르 코마로프"),
        ])
        self.assertEqual([n.name for n in nodes].count("블라디미르 코마로프"), 3)


class SourcedTieBreakTests(unittest.TestCase):
    def _row(self, uuid, name, label, ext=()):
        return {"uuid": uuid, "name": name, "labels": ["Entity", label], "same_label": True,
                "external_ids": list(ext)}

    def test_source_node_wins_over_agent_namesakes(self):
        rows = [self._row("agent", "헬무트 클로츠", "Person"),
                self._row("src", "헬무트 클로츠", "Person", ["commulingo:person:helmuth-klotz"]),
                self._row("agent2", "헬무트 클로츠", "Person")]
        self.assertEqual(identity._pick_key_hit(rows, "Person", "헬무트 클로츠").uuid, "src")

    def test_alias_tie_prefers_source_node(self):
        rows = [self._row("lenin", "Vladimir Lenin", "Person", ["commulingo:person:lenin"]),
                self._row("agent", "레닌", "Person")]
        self.assertEqual(identity._pick_key_hit(rows, "Person", "블라디미르 레닌").uuid, "lenin")

    def test_two_sources_or_cross_label_source_stay_ambiguous(self):
        rows = [self._row("a", "X", "Person", ["commulingo:person:a"]),
                self._row("b", "X", "Person", ["commulingo:person:b"])]
        self.assertEqual(identity._pick_key_hit(rows, "Person", "X").method, "ambiguous")
        rows = [dict(self._row("concept", "디아마트 (변증법적 유물론)", "Concept", ["commulingo:term:diamat"]), same_label=False),
                dict(self._row("org", "디아마트 (DiaMat)", "Organization"))]
        self.assertEqual(identity._pick_key_hit(rows, "Organization", "diamat").method, "ambiguous")


if __name__ == "__main__":
    unittest.main()
