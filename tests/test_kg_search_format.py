"""Hermetic tests for the KG read path helpers (kg_runtime.search / recall).

No Neo4j: covers tier parsing (both episode-name forms), source labelling,
edge/node line formatting, entity-mode dispatch with patched helpers, and
the flag-gated recall block.
"""

import os
import sys
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from kg_runtime import search as kgs  # noqa: E402
from kg_runtime import recall  # noqa: E402
from kg_runtime.identity import AliasHit  # noqa: E402


class TierAndSourceTests(unittest.TestCase):
    def test_tier_regex_accepts_both_prefixes(self):
        self.assertEqual(kgs._tier_from_names(["[T:anchor]structured-2026-x"]), "anchor")
        self.assertEqual(kgs._tier_from_names(["T-corroborated-scout-patrol"]), "corroborated")
        self.assertEqual(kgs._tier_from_names(["T-single-x", "[T:anchor]y"]), "anchor")  # best wins
        self.assertIsNone(kgs._tier_from_names(["scout-patrol-20260301"]))
        self.assertIsNone(kgs._tier_from_names(None))

    def test_source_label(self):
        self.assertEqual(kgs._source_label({"sync_key": "commulingo:event_person:a:b"}), "commulingo")
        self.assertEqual(kgs._source_label({"doc_ref": "research:my-slug"}), "research:my-slug")
        self.assertEqual(kgs._source_label({"ep_sources": ["agent_structured_write (scout)"]}), "scout")
        self.assertEqual(kgs._source_label({"ep_sources": ["Open source news article"]}), "news")
        self.assertEqual(kgs._source_label({}), "?")

    def test_label_of(self):
        self.assertEqual(kgs._label_of(["Entity", "Person"]), "Person")
        self.assertEqual(kgs._label_of(["Entity_economy", "Entity", "Asset"]), "Asset")
        self.assertEqual(kgs._label_of(["Entity"]), "Entity")


class FormatTests(unittest.TestCase):
    def test_edge_line_full(self):
        line = kgs._format_edge_line({
            "tier": "anchor", "expired_at": None, "subject": "니키타 흐루쇼프", "predicate": "Affiliation",
            "object": "당 지도부", "fact": "제1서기", "valid_at": "1953-01-01T00:00:00Z",
            "invalid_at": "1964-10-01T00:00:00Z", "source": "commulingo",
        })
        self.assertEqual(line, "- [anchor] 니키타 흐루쇼프 —Affiliation→ 당 지도부: 제1서기 (valid 1953-01-01 → 1964-10-01; src: commulingo)")

    def test_edge_line_expired_and_unknown(self):
        line = kgs._format_edge_line({"tier": None, "expired_at": "2026-05-01T00:00:00Z", "subject": "A",
                                      "predicate": "Statement", "object": "B", "fact": "said", "source": "?"})
        self.assertEqual(line, "- [?|expired] A —Statement→ B: said")

    def test_node_line(self):
        line = kgs._format_node_line({
            "name": "니키타 흐루쇼프", "labels": ["Entity", "Person"], "aliases": ["Nikita Khrushchev", "Хрущёв", "extra"],
            "external_ids": ["commulingo:person:khrushchev"], "summary": "x" * 300,
        })
        self.assertTrue(line.startswith("- 니키타 흐루쇼프 [Person] (aka: Nikita Khrushchev, Хрущёв; id: commulingo:person:khrushchev): "))
        self.assertTrue(line.endswith("…"))
        self.assertLess(len(line), 330)

    def test_format_results_falls_back_to_bare_fact_and_legacy_tier_map(self):
        out = kgs._format_kg_results(
            [{"name": "X", "labels": ["Entity", "Concept"], "summary": ""}],
            [{"uuid": "e1", "fact": "bare fact only"}],
            {"e1": "single"},
            entity_header="[hdr]",
        )
        self.assertEqual(out.splitlines(), [
            "[hdr]", "[Knowledge Graph: Entities]", "- X [Concept]",
            "[Knowledge Graph: Facts/Relations]", "- [single] bare fact only",
        ])


class EntityModeDispatchTests(unittest.TestCase):
    def _node(self):
        return {"uuid": "u1", "name": "니키타 흐루쇼프", "labels": ["Entity", "Person"], "summary": "요약",
                "aliases": ["Nikita Khrushchev"], "external_ids": ["commulingo:person:khrushchev"]}

    def _edges(self):
        return [
            {"uuid": "e1", "tier": "anchor", "expired_at": None, "subject": "니키타 흐루쇼프", "predicate": "Involvement",
             "object": "비밀연설", "fact": "연설했다", "valid_at": "1956-02-25", "invalid_at": None, "source": "commulingo"},
            {"uuid": "e2", "tier": None, "expired_at": "2026-01-01", "subject": "니키타 흐루쇼프", "predicate": "Statement",
             "object": "X", "fact": "old", "source": "scout"},
        ]

    def test_single_alias_hit_uses_entity_view_without_service(self):
        hit = AliasHit("u1", "니키타 흐루쇼프", ["Person"], "흐루쇼프")
        with mock.patch.object(kgs, "_alias_hits", return_value=[hit]), \
             mock.patch.object(kgs, "_entity_neighborhood", return_value=(self._node(), self._edges())), \
             mock.patch.object(kgs, "get_kg_service", side_effect=AssertionError("must not be called")):
            out = kgs.search_knowledge_graph("흐루쇼프가 비밀연설을 했나?")
        self.assertIn("[Knowledge Graph: entity view — 니키타 흐루쇼프 (1 active fact(s), 1 expired; matched via '흐루쇼프')]", out)
        self.assertIn("- [anchor] 니키타 흐루쇼프 —Involvement→ 비밀연설: 연설했다 (valid 1956-02-25; src: commulingo)", out)
        self.assertIn("- [?|expired] 니키타 흐루쇼프 —Statement→ X: old (src: scout)", out)

    def test_semantic_mode_skips_alias_index(self):
        fake_svc = mock.Mock()
        with mock.patch.object(kgs, "_alias_hits", side_effect=AssertionError("must not be called")), \
             mock.patch.object(kgs, "get_kg_service", return_value=fake_svc), \
             mock.patch.object(kgs, "run_kg_task", return_value={"nodes": [{"uuid": "n1", "name": "N", "labels": ["Entity", "Concept"], "summary": ""}], "edges": [{"uuid": "e9", "fact": "f"}]}), \
             mock.patch.object(kgs, "_hydrate_nodes", return_value={}), \
             mock.patch.object(kgs, "_hydrate_edges", return_value={"e9": {"uuid": "e9", "tier": "single", "expired_at": None, "subject": "N", "predicate": "Statement", "object": "M", "fact": "f", "source": "analyst"}}):
            out = kgs.search_knowledge_graph("anything", mode="semantic")
        self.assertIn("- N [Concept]", out)
        self.assertIn("- [single] N —Statement→ M: f (src: analyst)", out)

    def test_entity_mode_without_hit_returns_none(self):
        with mock.patch.object(kgs, "_resolve_entity_arg", return_value=None):
            self.assertIsNone(kgs.search_knowledge_graph("q", entity="Nobody", mode="entity"))

    def test_multiple_hits_add_mini_views_to_semantic(self):
        hits = [AliasHit("u1", "A", ["Person"], "a"), AliasHit("u2", "B", ["Person"], "b")]
        node_a = {"uuid": "u1", "name": "A", "labels": ["Entity", "Person"], "summary": "", "aliases": [], "external_ids": []}
        with mock.patch.object(kgs, "_alias_hits", return_value=hits), \
             mock.patch.object(kgs, "get_kg_service", return_value=mock.Mock()), \
             mock.patch.object(kgs, "run_kg_task", return_value={"nodes": [], "edges": []}), \
             mock.patch.object(kgs, "_entity_neighborhood", side_effect=lambda uuid, cap: (dict(node_a, uuid=uuid, name=uuid.upper()), [])):
            out = kgs.search_knowledge_graph("A and B")
        self.assertIn("entity view — U1", out)
        self.assertIn("entity view — U2", out)


class RecallTests(unittest.TestCase):
    def test_disabled_by_default(self):
        with mock.patch.dict(os.environ, {"KG_ENTITY_GATED_RECALL": "0"}):
            self.assertEqual(recall.entity_gated_kg_block("흐루쇼프", "claude"), "")

    def test_block_rendering(self):
        hit = AliasHit("u1", "니키타 흐루쇼프", ["Person"], "흐루쇼프")
        node = {"uuid": "u1", "name": "니키타 흐루쇼프", "summary": "요약", "labels": ["Entity", "Person"]}
        edges = [{"uuid": "e1", "tier": "anchor", "expired_at": None, "subject": "니키타 흐루쇼프", "predicate": "Involvement",
                  "object": "비밀연설", "fact": "연설했다", "source": "commulingo"}]
        with mock.patch.dict(os.environ, {"KG_ENTITY_GATED_RECALL": "1"}), \
             mock.patch.object(kgs, "_alias_hits", return_value=[hit]), \
             mock.patch.object(kgs, "_entity_neighborhood", return_value=(node, edges)):
            claude = recall.entity_gated_kg_block("흐루쇼프 이야기", "claude")
            openai = recall.entity_gated_kg_block("흐루쇼프 이야기", "openai")
        self.assertTrue(claude.startswith("<knowledge-graph>\n- 니키타 흐루쇼프: 요약\n  - [anchor] 니키타 흐루쇼프 —Involvement→ 비밀연설: 연설했다 (src: commulingo)"))
        self.assertTrue(claude.endswith("</knowledge-graph>"))
        self.assertTrue(openai.startswith("### Knowledge Graph"))

    def test_no_hit_is_empty(self):
        with mock.patch.dict(os.environ, {"KG_ENTITY_GATED_RECALL": "1"}), \
             mock.patch.object(kgs, "_alias_hits", return_value=[]):
            self.assertEqual(recall.entity_gated_kg_block("아무 이름 없음", "claude"), "")

    def test_reference_edges_are_not_rendered(self):
        hit = AliasHit("u1", "노동당", ["Organization"], "노동당")
        node = {"uuid": "u1", "name": "노동당", "summary": "", "labels": ["Entity", "Organization"]}
        ref = {"uuid": "e0", "tier": "note", "expired_at": None, "subject": "문서 A", "predicate": "Reference",
               "object": "노동당", "fact": "문서 'A'에 노동당이 언급된다", "source": "documents"}
        real = {"uuid": "e1", "tier": "anchor", "expired_at": None, "subject": "노동당", "predicate": "Statement",
                "object": "사회주의", "fact": "강령", "source": "analyst"}
        with mock.patch.dict(os.environ, {"KG_ENTITY_GATED_RECALL": "1"}), \
             mock.patch.object(kgs, "_alias_hits", return_value=[hit]), \
             mock.patch.object(kgs, "_entity_neighborhood", return_value=(node, [ref, ref, real, ref])):
            block = recall.entity_gated_kg_block("노동당 강령", "claude")
        self.assertIn("—Statement→", block)
        self.assertNotIn("Reference", block)
        self.assertNotIn("언급된다", block)

    def test_recall_asks_for_non_broad_hits(self):
        with mock.patch.dict(os.environ, {"KG_ENTITY_GATED_RECALL": "1"}), \
             mock.patch.object(kgs, "_alias_hits", return_value=[]) as hits:
            recall.entity_gated_kg_block("사회주의는 왜", "claude")
        self.assertFalse(hits.call_args.kwargs.get("broad", True))

    def test_mention_only_node_is_skipped(self):
        hit = AliasHit("u1", "국세청", ["Organization"], "국세청")
        node = {"uuid": "u1", "name": "국세청", "summary": "", "labels": ["Entity", "Organization"]}
        edges = [{"uuid": "e1", "tier": "note", "expired_at": None, "subject": "문서 A", "predicate": "Reference",
                  "object": "국세청", "fact": "문서 'A'에 국세청이 언급된다", "source": "documents"}]
        with mock.patch.dict(os.environ, {"KG_ENTITY_GATED_RECALL": "1"}), \
             mock.patch.object(kgs, "_alias_hits", return_value=[hit]), \
             mock.patch.object(kgs, "_entity_neighborhood", return_value=(node, edges)):
            self.assertEqual(recall.entity_gated_kg_block("국세청 조사", "claude"), "")


if __name__ == "__main__":
    unittest.main()
