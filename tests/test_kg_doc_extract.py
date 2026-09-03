"""Hermetic tests for kg_runtime.doc_extract (no DB, no Neo4j, no LLM)."""

import os
import sys
import unittest

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
