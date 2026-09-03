"""Hermetic tests for the CommuLingo → KG mapping (jobs/kg_sync_commulingo).

Feeds in-memory rows shaped like the Postgres tables and checks the facts
that come out: external ids, aliases, predicates, sync keys, dated
affiliations, incremental filtering, and validation against the writer's
schema gate with allow_sync_predicates=True.
"""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from jobs import kg_sync_commulingo as sync  # noqa: E402
from graph_memory.structured_writer import validate_fact  # noqa: E402


def _source():
    return sync.Source(
        people=[
            {"id": "khrushchev", "group_id": "thaw", "cyrillic": "Никита Хрущёв", "years_label": "1894–1971",
             "name_ko": "니키타 흐루쇼프", "name_en": "Nikita Khrushchev", "epithet_ko": "관 뚜껑을 연 사람",
             "bio_ko": "1956년 비밀연설로 관 뚜껑을 열었다.", "fate_kind": "deposed", "fate_label_ko": "실각"},
            {"id": "yezhov", "group_id": "stalin-era", "name_ko": "니콜라이 예조프", "name_en": "Nikolai Yezhov"},
            {"id": "lonely", "group_id": None, "name_ko": "외톨이", "name_en": "Loner"},
        ],
        person_aliases=[{"person_id": "khrushchev", "lang": "en", "alias": "Khrushchev"},
                        {"person_id": "khrushchev", "lang": "ko", "alias": "흐루쇼프"}],
        career=[{"person_id": "khrushchev", "sort_order": 0, "period_label": "1953–64", "role_ko": "제1서기"},
                {"person_id": "khrushchev", "sort_order": 1, "period_label": "1958–64", "role_ko": "각료회의 의장"}],
        person_roles=[{"person_id": "khrushchev", "office_id": "party-leadership", "label_ko": "제1서기", "category_id": None},
                      {"person_id": "yezhov", "office_id": "state-security", "label_ko": "NKVD 수장", "category_id": None},
                      {"person_id": "lonely", "office_id": None, "label_ko": "", "category_id": None}],
        role_categories=[{"id": "theorist", "label_ko": "이론가", "label_en": "Theorist"}],
        people_groups=[{"id": "thaw", "range_label": "1953–1985", "title_ko": "해빙과 정체의 사람들", "title_en": "People of the thaw"},
                       {"id": "stalin-era", "range_label": "1929–1953", "title_ko": "스탈린 시대의 사람들", "title_en": "Stalin era"}],
        offices=[{"id": "party-leadership", "title_ko": "당 지도부", "title_en": "Party leadership", "blurb_ko": "서기장 계보"},
                 {"id": "state-security", "title_ko": "국가보안 기관", "title_en": "State security agencies"}],
        office_rows=[{"id": 2, "office_id": "party-leadership", "period_label": "1953–1964", "start_year": 1953,
                      "start_month": None, "end_year": 1964, "end_month": 10, "body_ko": "제1서기 · 각료회의 의장",
                      "person_id": "khrushchev"}],
        events=[{"id": "great-terror", "period_label": "1936–1938", "title_ko": "대숙청", "title_en": "The Great Terror",
                 "summary_ko": "대규모 정치적 탄압.", "locations": [{"lat": 55.7, "lng": 37.6, "kind": "main",
                                                                 "label": {"en": "Moscow", "ko": "모스크바"}}]}],
        event_people=[{"event_id": "great-terror", "person_id": "yezhov", "relation_ko": "집행 총책",
                       "relation_en": "Operational chief", "note_ko": "NKVD 수장으로 지휘했다.", "relation_kind": "executor"}],
        terms=[{"id": "secret-speech", "term_ko": "비밀연설", "term_en": "Secret Speech", "original": "Секретный доклад",
                "period_label": "1956", "definition_ko": "제20차 당대회 연설.", "category": "party-state", "parent_id": None},
               {"id": "thaw-term", "term_ko": "해빙", "term_en": "Thaw", "category": "party-state", "parent_id": "secret-speech"}],
        term_aliases=[{"term_id": "secret-speech", "lang": "ko", "alias": "흐루쇼프 비밀연설"}],
        term_categories=[{"id": "party-state", "label_ko": "당·국가 기구", "label_en": "Party and state"}],
        term_relations=[{"term_id": "thaw-term", "related_id": "secret-speech"}],
        term_people=[{"term_id": "secret-speech", "person_id": "khrushchev"}],
        term_events=[{"term_id": "secret-speech", "event_id": "great-terror", "same_subject": False}],
        redirects=[],
    )


class SideTests(unittest.TestCase):
    def test_person_side(self):
        src = _source()
        ps = src.person("khrushchev")
        self.assertEqual(ps["name"], "니키타 흐루쇼프")
        self.assertEqual(ps["external_id"], "commulingo:person:khrushchev")
        for alias in ("Nikita Khrushchev", "Никита Хрущёв", "Khrushchev", "흐루쇼프"):
            self.assertIn(alias, ps["aliases"])
        self.assertNotIn("니키타 흐루쇼프", ps["aliases"])
        self.assertIn("1894–1971", ps["summary"])
        self.assertIn("주요 경력: 1953–64 제1서기; 1958–64 각료회의 의장", ps["summary"])
        self.assertIn("최후: 실각", ps["summary"])
        self.assertEqual(ps["name_en"], "Nikita Khrushchev")

    def test_location_side_slug(self):
        ls = sync.location_side({"label": {"en": "Panjshir Valley", "ko": "판지시르 계곡"}})
        self.assertEqual(ls["name"], "판지시르 계곡")
        self.assertEqual(ls["external_id"], "commulingo:location:panjshir-valley")
        self.assertEqual(ls["aliases"], ["Panjshir Valley"])
        self.assertIsNone(sync.location_side({"label": {}}))

    def test_josa(self):
        self.assertEqual(sync.josa("레닌", "은/는"), "은")
        self.assertEqual(sync.josa("흐루쇼프", "은/는"), "는")
        self.assertEqual(sync.josa("대숙청", "과/와"), "과")
        self.assertEqual(sync.josa("", "은/는"), "는")

    def test_year_dates(self):
        self.assertEqual(sync._year_date(1953), "1953-01-01")
        self.assertEqual(sync._year_date(1964, 10, end=True), "1964-10-01")
        self.assertEqual(sync._year_date(1964, None, end=True), "1964-12-31")
        self.assertIsNone(sync._year_date(None))


class BuildFactsTests(unittest.TestCase):
    def setUp(self):
        self.src = _source()
        self.facts = sync.build_facts(self.src)
        self.by_key = {f["attributes"]["sync_key"]: f for f in self.facts}

    def test_all_facts_pass_sync_validation_and_fail_agent_validation_for_reference(self):
        for i, f in enumerate(self.facts):
            self.assertIsNone(validate_fact(f, i, allow_sync_predicates=True), f)
        refs = [f for f in self.facts if f["predicate"] == "Reference"]
        self.assertTrue(refs)
        self.assertIsNotNone(validate_fact(refs[0], 0))

    def test_sync_keys_unique(self):
        keys = [f["attributes"]["sync_key"] for f in self.facts]
        self.assertEqual(len(keys), len(set(keys)))

    def test_every_person_has_at_least_one_edge(self):
        for pid in self.src.people:
            ext = f"commulingo:person:{pid}"
            self.assertTrue(any(
                f.get("subject_external_id") == ext or f.get("object_external_id") == ext for f in self.facts
            ), pid)
        # person with no group/role/office gets the collection fallback
        self.assertIn("commulingo:person_collection:lonely", self.by_key)

    def test_office_row_dated_affiliation(self):
        f = self.by_key["commulingo:office_row:2"]
        self.assertEqual((f["subject_type"], f["predicate"], f["object_type"]), ("Person", "Affiliation", "Role"))
        self.assertEqual(f["object_external_id"], "commulingo:office:party-leadership")
        self.assertEqual(f["valid_at"], "1953-01-01")
        self.assertEqual(f["invalid_at"], "1964-10-01")
        self.assertEqual(f["attributes"]["position"], "제1서기 · 각료회의 의장")

    def test_event_people_and_location(self):
        inv = self.by_key["commulingo:event_person:great-terror:yezhov"]
        self.assertEqual((inv["subject_name"], inv["predicate"], inv["object_name"]), ("니콜라이 예조프", "Involvement", "대숙청"))
        self.assertEqual(inv["attributes"]["role_in_incident"], "executor")
        self.assertIn("집행 총책", inv["fact"])
        self.assertEqual(inv["object_summary"], "1936–1938. 대규모 정치적 탄압.")
        loc = self.by_key["commulingo:event_location:great-terror:moscow"]
        self.assertEqual((loc["subject_type"], loc["predicate"], loc["object_type"]), ("Incident", "Presence", "Location"))
        self.assertEqual(loc["attributes"]["presence_type"], "main")

    def test_term_links(self):
        cat = self.by_key["commulingo:term_category:secret-speech"]
        self.assertEqual(cat["attributes"]["reference_type"], "category")
        self.assertEqual(cat["object_external_id"], "commulingo:term-category:party-state")
        parent = self.by_key["commulingo:term_parent:thaw-term"]
        self.assertEqual(parent["object_name"], "비밀연설")
        rel = self.by_key["commulingo:term_relation:thaw-term:secret-speech"]
        self.assertEqual(rel["attributes"]["reference_type"], "related_term")
        tp = self.by_key["commulingo:term_person:secret-speech:khrushchev"]
        self.assertEqual((tp["subject_type"], tp["object_type"]), ("Person", "Concept"))
        self.assertIn("흐루쇼프 비밀연설", tp["object_aliases"])
        te = self.by_key["commulingo:term_event:secret-speech:great-terror"]
        self.assertEqual((te["subject_type"], te["object_type"]), ("Concept", "Incident"))
        self.assertIs(te["attributes"]["same_subject"], False)

    def test_people_group_reference(self):
        f = self.by_key["commulingo:person_group:khrushchev"]
        self.assertEqual(f["object_external_id"], "commulingo:people-group:thaw")
        self.assertEqual(f["object_name"], "해빙과 정체의 사람들")
        self.assertIn("니키타 흐루쇼프는 CommuLingo 인물사전의 '해빙과 정체의 사람들' (1953–1985)", f["fact"])
        self.assertEqual(f["attributes"]["reference_type"], "people_group")

    def test_incremental_filter(self):
        facts = sync.build_facts(self.src, changed={"person": {"yezhov"}, "event": set(), "term": set(), "office": set()})
        keys = {f["attributes"]["sync_key"] for f in facts}
        self.assertIn("commulingo:event_person:great-terror:yezhov", keys)
        self.assertIn("commulingo:person_office:yezhov:state-security", keys)
        self.assertNotIn("commulingo:office_row:2", keys)
        self.assertNotIn("commulingo:term_category:secret-speech", keys)
        self.assertEqual(sync.build_facts(self.src, changed={"person": set(), "event": set(), "term": set(), "office": set()}), [])


class RunDiffTests(unittest.TestCase):
    def test_run_dry_run_diff_logic(self):
        """run() with dry_run computes the diff without touching Neo4j/Postgres writes."""
        src = _source()
        orig_load = sync.load_source
        sync.load_source = lambda: src
        try:
            stats = sync.run(full=True, dry_run=True)
        finally:
            sync.load_source = orig_load
        self.assertEqual(stats["facts_total"], stats["facts_new_or_changed"])
        self.assertEqual(stats["edges_to_expire"], 0)
        self.assertTrue(stats["sample"])
        self.assertEqual(stats["source_rows"]["people"], 3)


if __name__ == "__main__":
    unittest.main()
