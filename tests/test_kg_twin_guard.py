"""Cross-script twin guard and rejection hints in the structured KG writer."""
import unittest
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from graph_memory import structured_writer as writer


class _Client:
    @asynccontextmanager
    async def session(self, **kwargs):
        yield object()


def _fact(subject, obj="DMZ 지뢰 폭발 사건", **extra):
    return {"subject_name": subject, "subject_type": "Person", "predicate": "Involvement",
            "object_name": obj, "object_type": "Incident", "fact": f"{subject} commented", **extra}


async def _embed(embedder, nodes, edges):
    for node in nodes:
        node.name_embedding = [0.1, 0.2]


async def _twin(session, name, etype, vec):
    if name == "Kim Yo Jong":
        return {"uuid": "old", "name": "김여정", "score": 0.87}
    return None


class TwinGuardTests(unittest.IsolatedAsyncioTestCase):
    async def _write(self, facts, guard=True):
        graph = SimpleNamespace(driver=SimpleNamespace(client=_Client()), embedder=object())
        with patch.object(writer, "find_canonical_entity_uuid", new=AsyncMock(return_value=None)), \
             patch.object(writer, "_untrusted_aliases", new=AsyncMock(return_value=[])), \
             patch.object(writer, "_embed_in_batches", side_effect=_embed), \
             patch.object(writer, "_cross_script_twin", side_effect=_twin), \
             patch.object(writer, "add_nodes_and_edges_bulk", new=AsyncMock()) as save, \
             patch.object(writer, "validate_episode_result", return_value=None):
            result = await writer.write_structured_facts(graph, facts, group_id="geopolitics_conflict",
                                                         cross_script_guard=guard)
        return result, save

    async def test_twin_is_rejected_with_existing_name_and_rest_written(self):
        result, save = await self._write([_fact("Kim Yo Jong"), _fact("강신철")])
        self.assertEqual(result["status"], "partial_success")
        self.assertEqual(result["written_fact_indices"], [1])
        reason = result["rejected_facts"][0]["reason"]
        self.assertIn("'김여정'", reason)
        self.assertIn("confirm_new_entities", reason)
        saved_names = {n.name for n in save.await_args.args[3]}
        self.assertNotIn("Kim Yo Jong", saved_names)
        self.assertIn("강신철", saved_names)

    async def test_confirmed_fact_and_disabled_guard_write_new_node(self):
        result, _ = await self._write([_fact("Kim Yo Jong", confirm_new_entities=True)])
        self.assertEqual(result["status"], "ok")
        result, _ = await self._write([_fact("Kim Yo Jong")], guard=False)
        self.assertEqual(result["status"], "ok")

    async def test_all_rejected_is_error_without_save(self):
        result, save = await self._write([_fact("Kim Yo Jong")])
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["facts_written"], 0)
        save.assert_not_awaited()

    def test_script_detection(self):
        self.assertTrue(writer._has_hangul("Hong Myeong-gyo (홍명교)"))
        self.assertFalse(writer._has_hangul("Kim Yo Jong"))


class RejectionHintTests(unittest.TestCase):
    def test_reverse_direction_hint(self):
        reason = writer.validate_fact({
            "subject_name": "은행권 해킹 사건", "subject_type": "Incident", "predicate": "Involvement",
            "object_name": "BNK부산은행", "object_type": "Organization", "fact": "x"}, 0)
        self.assertIn("IS allowed for (Organization -> Incident)", reason)



class AuditMetadataTests(unittest.IsolatedAsyncioTestCase):
    """Tool result_metadata must pass the audit sink's field allowlist, or the
    row is rejected (HTTP 400) and blocks the spool batch behind it (2026-10-06)."""

    async def test_write_and_retract_metadata_pass_the_sink(self):
        from ops import audit_sink
        from self_runtime import tools

        results = [
            {"status": "partial_success", "message": "m", "facts_written": 2, "facts_rejected": 1,
             "rejected_facts": [{"index": 1, "reason": "r", "fact": {}}]},
            {"status": "error", "message": "m", "facts_written": 0, "facts_rejected": 1, "rejected_facts": []},
        ]
        for result in results:
            with patch("kg_runtime.writes.add_kg_structured_async", new=AsyncMock(return_value=result)):
                out = await tools._exec_write_kg_structured([_fact("강신철")], group_id="korea_domestic")
            audit_sink.normalize_row("tool", {"tool_name": "write_kg_structured", "decision": "allow",
                                              "result_metadata": out.result_metadata})
        for status in ("retracted", "ambiguous", "refused", "not_found", "error"):
            with patch("kg_runtime.writes.retract_kg_fact", return_value={"status": status, "message": "m"}):
                out = await tools._exec_retract_kg_fact("a", "Statement", "b", "reason long enough here")
            audit_sink.normalize_row("tool", {"tool_name": "retract_kg_fact", "decision": "allow",
                                              "result_metadata": out.result_metadata})


if __name__ == "__main__":
    unittest.main()
