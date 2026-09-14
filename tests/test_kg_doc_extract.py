"""Hermetic tests for kg_runtime.doc_extract (no DB, no Neo4j, no LLM)."""

import os
import sys
import unittest
from unittest.mock import patch
from unittest.mock import MagicMock
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from kg_runtime import doc_extract as dx  # noqa: E402
from kg_runtime.identity import AliasIndex  # noqa: E402
from graph_memory.structured_writer import validate_fact  # noqa: E402


MANIFEST_DOC = {
    "id": "marx-engels-communist-manifesto", "file": "x.html", "docLang": "ko", "date": "1848-02",
    "title": {"ko": "마르크스·엥겔스, 『공산당 선언』", "en": "Marx & Engels, The Communist Manifesto"},
    "description": {"ko": "1848년 런던에서 발표한 문헌의 한국어 번역 전문.", "en": "..."},
    "kind": {"ko": "저작·연설", "en": "Writings & speeches"},
    "source": "Karl Marx & Friedrich Engels, Manifest der Kommunistischen Partei (London, 1848)",
    "people": ["karl-marx", "friedrich-engels", "unknown-slug"], "terms": ["scientific-socialism"], "events": [],
    "addedAt": "2026-08-05",
    "aliases": {"ko": ["『공산당 선언』", "공산당 선언"], "en": ["The Communist Manifesto"]},
}
NAMES = {"person": {"karl-marx": "카를 마르크스", "friedrich-engels": "프리드리히 엥겔스"},
         "term": {"scientific-socialism": "과학적 사회주의"}, "event": {}}


class RecordTests(unittest.TestCase):
    def test_archival_record(self):
        rec = dx.archival_record(MANIFEST_DOC, "<html><body><h1>공산당 선언</h1><p>부르주아지와 프롤레타리아</p><script>x</script></body></html>")
        self.assertEqual(rec.ref, "archival:marx-engels-communist-manifesto")
        self.assertEqual(rec["title"], "마르크스·엥겔스, 『공산당 선언』")
        self.assertIn("공산당 선언", rec["text"])
        self.assertNotIn("script", rec["text"])
        self.assertIn("The Communist Manifesto", rec["aliases"])
        self.assertEqual(rec["links"]["person"], ["karl-marx", "friedrich-engels", "unknown-slug"])
        self.assertEqual(len(rec["sha"]), 64)
        side = dx.document_side(rec)
        self.assertEqual(side["type"], "Document")
        self.assertTrue(side["summary"].startswith("저작·연설 · 1848-02. "))

    def test_research_and_autonote_records(self):
        rec = dx.research_record({"slug": "s", "title": "T", "markdown": "# T\n\n**bold** [link](http://x) body",
                                  "summary": None, "tags": '["a"]', "published_at": "2026-08-11 02:46:54+00",
                                  "content_sha256": "abc", "lang": "ko"})
        self.assertEqual(rec.ref, "research:s")
        self.assertEqual(rec["published_at"], "2026-08-11")
        self.assertIn("bold link body", rec["text"])
        self.assertEqual(rec["tags"], ["a"])
        note = dx.autonote_record({"id": 7, "project_id": 3, "text": "# 2분기 마감\n본문", "created_at": "2026-07-26", "kind": "synthesis"})
        self.assertEqual((note.ref, note["title"]), ("autonote:7", "2분기 마감"))


class FactTests(unittest.TestCase):
    def setUp(self):
        self.rec = dx.archival_record(MANIFEST_DOC, "<p>레닌은 이 문헌을 읽었다. 카를 마르크스의 저작.</p>")
        self.idx = AliasIndex()
        self.idx.load_rows([
            {"uuid": "u-lenin", "name": "블라디미르 레닌", "labels": ["Entity", "Person"], "keys": ["레닌"]},
            {"uuid": "u-marx", "name": "카를 마르크스", "labels": ["Entity", "Person"], "keys": []},
            {"uuid": "u-doc", "name": "어떤 문서", "labels": ["Entity", "Document"], "keys": ["문헌"]},
        ])

    def test_curated_links_skip_unknown_and_use_external_ids(self):
        facts = dx.curated_link_facts(self.rec, NAMES)
        self.assertEqual(len(facts), 3)
        ids = {f["object_external_id"] for f in facts}
        self.assertEqual(ids, {"commulingo:person:karl-marx", "commulingo:person:friedrich-engels", "commulingo:term:scientific-socialism"})
        f = facts[0]
        self.assertEqual((f["subject_type"], f["predicate"], f["object_type"]), ("Document", "Reference", "Person"))
        self.assertEqual(f["attributes"]["reference_type"], "about")
        self.assertEqual(f["attributes"]["doc_ref"], "archival:marx-engels-communist-manifesto")
        self.assertEqual(f["valid_at"], "2026-08-05")

    def test_mentions_skip_document_labels(self):
        facts = dx.mention_facts(self.rec, self.idx)
        names = {f["object_name"] for f in facts}
        self.assertIn("블라디미르 레닌", names)
        self.assertIn("카를 마르크스", names)
        self.assertNotIn("어떤 문서", names)
        self.assertTrue(all(f["attributes"]["reference_type"] == "mentions" for f in facts))

    def test_build_document_facts_validates_and_dedupes(self):
        facts = dx.build_document_facts(self.rec, names=NAMES, alias_index=self.idx, use_llm=False)
        keys = [f["attributes"]["sync_key"] for f in facts]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(facts[0]["attributes"]["reference_type"], "collection")
        for i, f in enumerate(facts):
            self.assertIsNone(validate_fact(f, i, allow_sync_predicates=True), f)
            self.assertIsNotNone(validate_fact(f, i))  # Document/Reference never pass the agent gate


class LLMParsingTests(unittest.TestCase):
    def test_self_loop_is_dropped_without_orphan_mention(self):
        rec = dx.research_record({"slug": "s", "title": "T", "markdown": "x"})
        loop = {"subject_name": "International Workingmen Association", "subject_type": "Organization",
                "predicate": "Statement", "object_name": "international workingmen association",
                "object_type": "Organization", "fact": "The association declared its principles."}
        valid = dict(loop, object_name="General Rules", object_type="Policy")
        facts = dx.llm_facts(rec, [loop, valid])
        extracted = [f for f in facts if f["attributes"].get("extraction") == "llm"]
        self.assertEqual(len(extracted), 1)
        self.assertEqual(extracted[0]["object_name"], "General Rules")
        self.assertEqual(len(facts), 3)
        self.assertEqual(dx.llm_facts(rec, [loop]), [])

    def test_same_name_different_types_is_not_assumed_self_loop(self):
        rec = dx.research_record({"slug": "s", "title": "T", "markdown": "x"})
        facts = dx.llm_facts(rec, [{"subject_name": "Lenin", "subject_type": "Person",
                                   "predicate": "Statement", "object_name": "Lenin",
                                   "object_type": "Concept", "fact": "A person discussed a titled work."}])
        self.assertEqual(len(facts), 3)

    RAW = '''```json
{"facts": [
  {"subject_name": "니키타 흐루쇼프", "subject_type": "Person", "predicate": "Statement", "object_name": "비밀연설",
   "object_type": "Concept", "fact": "흐루쇼프는 1956년 2월 25일 제20차 당대회에서 비밀연설을 했다.", "valid_at": "1956-02-25",
   "subject_aliases": ["Nikita Khrushchev", "니키타 흐루쇼프"], "object_aliases": ["Secret Speech"]},
  {"subject_name": "X", "subject_type": "Organization", "predicate": "Involvement", "object_name": "Y",
   "object_type": "Concept", "fact": "bad pair"},
  {"subject_name": "Doc", "subject_type": "Document", "predicate": "Reference", "object_name": "Z",
   "object_type": "Person", "fact": "never from the LLM"},
  "not a dict"
]}
```'''

    def test_parse_and_validate(self):
        rec = dx.research_record({"slug": "s", "title": "T", "markdown": "x", "published_at": "2026-08-11"})
        raw = dx.parse_llm_facts(self.RAW)
        self.assertEqual(len(raw), 3)
        facts = dx.llm_facts(rec, raw)
        llm = [f for f in facts if f["attributes"].get("extraction") == "llm"]
        mentions = [f for f in facts if f["attributes"].get("reference_type") == "mentions"]
        self.assertEqual(len(llm), 1)                       # bad pair + Document/Reference dropped
        self.assertEqual(llm[0]["attributes"]["doc_ref"], "research:s")
        self.assertEqual(llm[0]["valid_at"], "1956-02-25")
        self.assertEqual(llm[0]["subject_aliases"], ["Nikita Khrushchev"])  # self-alias removed
        self.assertEqual({m["object_name"] for m in mentions}, {"니키타 흐루쇼프", "비밀연설"})
        self.assertEqual(mentions[0]["subject_type"], "Document")

    def test_generic_entities_are_dropped(self):
        rec = dx.research_record({"slug": "s", "title": "T", "markdown": "x", "published_at": None})
        raw = [
            {"subject_name": "국가", "subject_type": "Concept", "predicate": "Causation",
             "object_name": "계급 적대", "object_type": "Concept", "fact": "국가는 계급 적대의 산물이다"},
            {"subject_name": "European Union", "subject_type": "Organization", "predicate": "PolicyEffect",
             "object_name": "Organization", "object_type": "Organization", "fact": "type label as a name"},
            {"subject_name": "카를 마르크스", "subject_type": "Person", "predicate": "Statement",
             "object_name": "국가와 혁명", "object_type": "Concept", "fact": "마르크스는 국가론을 썼다"},
        ]
        facts = dx.llm_facts(rec, raw)
        llm = [f for f in facts if f["attributes"].get("extraction") == "llm"]
        self.assertEqual([f["subject_name"] for f in llm], ["카를 마르크스"])
        self.assertNotIn("국가", {f["object_name"] for f in facts})
        self.assertIn("generic common noun", dx.EXTRACTION_SYSTEM)
        from graph_memory.structured_writer import validate_fact
        err = validate_fact({"subject_name": "정부", "subject_type": "Organization", "predicate": "Statement",
                             "object_name": "금리 인하", "object_type": "Policy", "fact": "정부가 말했다"}, 0)
        self.assertIn("generic noun", err or "")   # agent write_kg_structured path too

    def test_llm_sides_are_untrusted_and_mentions_skip_broad_keys(self):
        from graph_memory.structured_writer import _side
        rec = dx.research_record({"slug": "s", "title": "T", "markdown": "x", "published_at": None})
        raw = [{"subject_name": "소련", "subject_type": "Organization", "predicate": "OrgRelation",
                "object_name": "United States", "object_type": "Organization", "fact": "미소 데탕트",
                "subject_aliases": ["Soviet Union"], "object_aliases": ["미국"]}]
        facts = dx.llm_facts(rec, raw)
        llm = [f for f in facts if f["attributes"].get("extraction") == "llm"][0]
        self.assertFalse(_side(llm, "subject")["trusted"])
        self.assertFalse(_side(llm, "object")["trusted"])
        doc_side = [f for f in facts if f["subject_type"] == "Document"][0]
        self.assertTrue(_side(doc_side, "subject")["trusted"])   # document node carries its ref

        class _Idx:
            def __init__(self):
                self.kwargs = None

            def match(self, text, limit=5, **kw):
                self.kwargs = kw
                return []
        idx = _Idx()
        dx.mention_facts(rec, idx)
        self.assertEqual(idx.kwargs, {"broad": False})

    def test_parse_tolerates_bare_list_and_garbage(self):
        self.assertEqual(len(dx.parse_llm_facts('[{"a": 1}, 2]')), 1)
        self.assertEqual(dx.parse_llm_facts("no json here"), [])
        self.assertEqual(dx.parse_llm_facts(""), [])

    def test_prompt_caps_text(self):
        rec = dx.research_record({"slug": "s", "title": "T", "markdown": "y" * 50000, "published_at": None})
        prompt = dx.build_llm_prompt(rec)
        self.assertLess(len(prompt), dx.MAX_LLM_CHARS + 500)
        self.assertTrue(prompt.startswith("Document: T\nKind: research"))


class IdempotencyTests(unittest.TestCase):
    def test_alias_self_loop_filter_preserves_required_links_and_distinct_claims(self):
        loop = {"subject_name": "IWMA", "subject_type": "Organization", "object_name": "First International",
                "object_type": "Organization", "attributes": {"extraction": "llm", "sync_key": "doc:x:llm:0"}}
        other = dict(loop, object_name="General Rules", object_type="Policy")
        required = dict(loop, attributes={"reference_type": "about"})
        driver_context = MagicMock()
        driver_context.__enter__.return_value = (MagicMock(), "neo4j")
        def resolve(session, *, name, entity_type, trusted):
            self.assertFalse(trusted)
            return SimpleNamespace(uuid="rules" if name == "General Rules" else "iwma")
        with patch("kg_runtime.search._get_neo4j_sync_driver", return_value=driver_context), \
             patch("kg_runtime.identity.resolve_entity_sync", side_effect=resolve):
            kept, skipped = dx.filter_resolved_self_loops([loop, other, required])
        self.assertEqual(kept, [other, required])
        self.assertEqual(skipped[0]["sync_key"], "doc:x:llm:0")

    def test_unresolved_entities_are_not_assumed_identical(self):
        fact = {"subject_name": "Alpha", "subject_type": "Organization", "object_name": "Beta",
                "object_type": "Organization", "attributes": {"extraction": "llm"}}
        driver_context = MagicMock()
        driver_context.__enter__.return_value = (MagicMock(), "neo4j")
        with patch("kg_runtime.search._get_neo4j_sync_driver", return_value=driver_context), \
             patch("kg_runtime.identity.resolve_entity_sync", return_value=SimpleNamespace(uuid=None)):
            self.assertEqual(dx.filter_resolved_self_loops([fact]), ([fact], []))

    def test_identity_failure_does_not_write_or_stamp(self):
        rec = dx.research_record({"slug": "s", "title": "T", "markdown": "x"})
        fact = {"attributes": {"extraction": "llm"}}
        with patch.object(dx, "build_document_facts", return_value=[fact]), \
             patch("kg_runtime.search._get_neo4j_sync_driver", side_effect=RuntimeError("offline")), \
             patch.object(dx, "write_document_facts") as write, patch.object(dx, "stamp_document_node") as stamp:
            with self.assertRaisesRegex(RuntimeError, "offline"):
                dx.extract_document(rec)
        write.assert_not_called()
        stamp.assert_not_called()

    def test_partial_write_reports_rejection_without_stamping_or_expiring(self):
        rec = dx.research_record({"slug": "s", "title": "T", "markdown": "x"})
        response = {"status": "partial_success", "facts_written": 1, "facts_rejected": 1,
                    "edge_uuids": ["saved"], "message": "wrote facts " + "x" * 300,
                    "rejected_facts": [{"index": 1, "reason": "self-loop after entity resolution", "fact": {}}]}
        with patch.object(dx, "build_document_facts", return_value=[{}, {}]), \
             patch.object(dx, "write_document_facts", return_value=response), \
             patch.object(dx, "stamp_document_node") as stamp, \
             patch.object(dx, "expire_document_edges") as expire:
            result = dx.extract_document(rec)
        self.assertEqual(result["status"], "error")
        self.assertTrue(result["message"].startswith("fact[1]: self-loop after entity resolution"))
        self.assertLessEqual(len(result["message"]), 200)
        self.assertEqual(result["rejected_facts"], [{"index": 1, "reason": "self-loop after entity resolution"}])
        stamp.assert_not_called()
        expire.assert_not_called()

    def test_unchanged_hash_skips_without_touching_graph(self):
        rec = dx.research_record({"slug": "s", "title": "T", "markdown": "x", "content_sha256": "same"})
        res = dx.extract_document(rec, existing_sha="same", use_llm=False)
        self.assertEqual(res, {"ref": "research:s", "status": "unchanged"})

    def test_llm_flag_default_off(self):
        old = os.environ.pop("KG_DOC_EXTRACT_LLM", None)
        try:
            self.assertFalse(dx.llm_enabled())
            os.environ["KG_DOC_EXTRACT_LLM"] = "1"
            self.assertTrue(dx.llm_enabled())
        finally:
            os.environ.pop("KG_DOC_EXTRACT_LLM", None)
            if old is not None:
                os.environ["KG_DOC_EXTRACT_LLM"] = old


if __name__ == "__main__":
    unittest.main()
