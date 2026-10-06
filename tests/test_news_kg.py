"""Daily news → KG job: parsing, selection validation, fact normalisation, tier."""
import json
import unittest
from unittest.mock import patch

from jobs import news_kg
from web_gateway.search import _format_results

RUN_DATE = "2026-10-06"


def candidate(i, url, title="t", published="2026-10-05"):
    return {"title": f"{title}{i}", "url": url, "published": published, "snippet": "s",
            "url_hash": news_kg.url_hash(url), "domain": "경제·시장"}


class SearchParseTests(unittest.TestCase):
    def test_round_trips_gateway_format(self):
        text = _format_results("tavily", "q", [
            {"title": "Strike at Hyundai (x)", "url": "https://a.example/news/1?utm_source=x",
             "content": "Workers walked out.\nSecond line.", "published_date": "Mon, 05 Oct 2026"},
            {"title": "No date", "url": "https://b.example/2", "content": "Body"},
        ], advanced=False)
        items = news_kg.parse_search_results(text)
        self.assertEqual([i["url"] for i in items],
                         ["https://a.example/news/1?utm_source=x", "https://b.example/2"])
        self.assertEqual(items[0]["title"], "Strike at Hyundai (x)")
        self.assertEqual(items[0]["published"], "Mon, 05 Oct 2026")
        self.assertEqual(items[0]["snippet"], "Workers walked out.\nSecond line.")
        self.assertEqual(items[1]["published"], "")
        self.assertNotIn("</external>", items[1]["snippet"])

    def test_empty_result_parses_to_nothing(self):
        self.assertEqual(news_kg.parse_search_results(_format_results("tavily", "q", [], advanced=False)), [])

    def test_url_hash_ignores_tracking_and_fragment(self):
        self.assertEqual(news_kg.url_hash("https://WWW.a.com/x/?utm_medium=y#top"),
                         news_kg.url_hash("https://www.a.com/x"))
        self.assertNotEqual(news_kg.url_hash("https://a.com/x?id=1"), news_kg.url_hash("https://a.com/x?id=2"))


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.cands = [candidate(0, "https://a.com/1"), candidate(1, "https://www.a.com/2"),
                      candidate(2, "https://b.com/3"), candidate(3, "https://c.com/4")]

    def test_validates_groups_indices_and_domains(self):
        raw = json.dumps({"stories": [
            {"title": "파업", "group_id": "korea_domestic", "items": [0, 1, 2, 99, "x"]},
            {"title": "bad group", "group_id": "labor", "items": [3]},
            {"title": "reused", "group_id": "economy", "items": [0]},
            {"title": "ok", "group_id": "economy", "items": [3]},
        ]})
        stories = news_kg.parse_selection(raw, self.cands, max_stories=6)
        self.assertEqual([s["title"] for s in stories], ["파업", "ok"])
        # a.com and www.a.com are one outlet: only the first is kept; b.com corroborates.
        self.assertEqual([c["url"] for c in stories[0]["candidates"]], ["https://a.com/1", "https://b.com/3"])

    def test_caps_story_count(self):
        raw = json.dumps({"stories": [{"title": str(i), "group_id": "economy", "items": [i]} for i in range(4)]})
        self.assertEqual(len(news_kg.parse_selection("```json\n" + raw + "\n```", self.cands, max_stories=2)), 2)


class FactTests(unittest.TestCase):
    def story(self):
        return {"title": "현대차 파업", "story_key": "k1", "group_id": "korea_domestic",
                "articles": [{**candidate(0, "https://a.com/1"), "text": "x"},
                             {**candidate(1, "https://b.com/2", published=""), "text": "y"}]}

    def test_normalises_dates_provenance_and_drops_bad_facts(self):
        raw = [
            {"subject_name": "전국금속노동조합", "subject_type": "Organization", "predicate": "Involvement",
             "object_name": "2026년 현대자동차 파업", "object_type": "Incident", "fact": "금속노조가 파업했다.",
             "valid_at": "2026-10-04", "invalid_at": "2026-10-01", "article": 2},
            {"subject_name": "현대자동차", "subject_type": "Organization", "predicate": "Statement",
             "object_name": "2026년 현대자동차 파업", "object_type": "Incident", "fact": "회사가 유감을 표했다.",
             "valid_at": "2026-12-01", "article": 7},
            {"subject_name": "Donald Trump", "subject_type": "Person", "predicate": "Statement",
             "object_name": "알래스카 LNG 사업", "object_type": "Campaign", "fact": "트럼프가 밝혔다.",
             "valid_at": "2025-09-30"},
            {"subject_name": "정부", "subject_type": "Organization", "predicate": "Statement",
             "object_name": "2026년 현대자동차 파업", "object_type": "Incident", "fact": "정부가 말했다."},
        ]
        facts, rejected = news_kg.normalize_facts(self.story(), raw, RUN_DATE)
        self.assertEqual(len(facts), 3)  # generic "정부" goes to the repair round
        self.assertEqual(len(rejected), 1)
        self.assertIn("generic", rejected[0]["reason"])
        first, second, third = facts
        # Dates are the model's call: scheduled (future) and past dates both pass through.
        self.assertEqual(second["valid_at"], "2026-12-01")
        self.assertEqual(third["valid_at"], "2025-09-30")
        self.assertEqual(first["valid_at"], "2026-10-04")
        self.assertNotIn("invalid_at", first)  # end before start
        self.assertEqual(first["attributes"]["source_url"], "https://b.com/2")
        self.assertNotIn("news_published", first["attributes"])
        self.assertEqual(second["attributes"]["news_published"], "2026-10-05")
        self.assertEqual(second["attributes"]["source_url"], "https://a.com/1")  # bad article number
        for f in facts:
            self.assertNotIn("sync_key", f["attributes"])  # stays retractable
            self.assertEqual(f["attributes"]["story_key"], "k1")

    def test_publication_dates_become_iso_and_unknown_is_anchored(self):
        self.assertEqual(news_kg.published_date("Mon, 05 Oct 2026 22:10:00 GMT"), "2026-10-05")
        self.assertEqual(news_kg.published_date("2026-10-05T01:00:00Z"), "2026-10-05")
        self.assertIsNone(news_kg.published_date("3 hours ago"))
        self.assertIn(RUN_DATE, news_kg.build_extract_prompt(self.story(), RUN_DATE))
        self.assertIn("Tuesday", news_kg.today_label(RUN_DATE))

    def test_reversed_direction_is_repaired_by_the_model(self):
        story = self.story()
        reversed_fact = {"subject_name": "2026년 현대자동차 파업", "subject_type": "Incident",
                         "predicate": "Involvement", "object_name": "전국금속노동조합",
                         "object_type": "Organization", "fact": "금속노조가 파업했다.", "article": 2}
        fixed = {**reversed_fact, "subject_name": "전국금속노동조합", "subject_type": "Organization",
                 "object_name": "2026년 현대자동차 파업", "object_type": "Incident"}
        replies = [json.dumps({"facts": [reversed_fact]}), json.dumps({"facts": [fixed]})]
        with patch("llm.call_registry.generate_sync", side_effect=replies) as gen, \
                patch.object(news_kg.dx, "filter_resolved_self_loops", side_effect=lambda f: (f, [])):
            facts = news_kg.extract_story_facts(story, RUN_DATE)
        self.assertEqual(gen.call_count, 2)
        repair_prompt = gen.call_args_list[1].args[1]
        self.assertIn("swap subject and object", repair_prompt)
        self.assertIn('"article": 2', repair_prompt)
        self.assertEqual([(f["subject_name"], f["object_name"]) for f in facts],
                         [("전국금속노동조합", "2026년 현대자동차 파업")])
        self.assertEqual(facts[0]["attributes"]["source_url"], "https://b.com/2")

    def test_writer_rejection_gets_one_repair_and_rewrite(self):
        story = self.story()
        fact = {"subject_name": "키이우", "subject_type": "Location", "predicate": "Presence",
                "object_name": "Ukraine", "object_type": "Organization", "fact": "x",
                "attributes": {"source_url": "https://a.com/1"}}
        first = {"status": "error", "facts_written": 0, "facts_rejected": 1, "message": "no facts written",
                 "rejected_facts": [{"index": 0, "reason": "the KG already has 'Kyiv'", "fact": fact}]}
        second = {"status": "ok", "facts_written": 1, "facts_rejected": 0}
        repaired = [{**fact, "subject_name": "Kyiv"}]
        with patch.object(news_kg, "_write", side_effect=[first, second]) as write, \
                patch.object(news_kg, "repair_facts", return_value=repaired) as repair:
            res = news_kg.write_story(story, [fact], RUN_DATE)
        self.assertEqual(repair.call_args.args[1][0]["reason"], "the KG already has 'Kyiv'")
        self.assertEqual(write.call_args_list[1].args[1], repaired)
        self.assertEqual((res["status"], res["facts_written"], res["facts_rejected"]), ("ok", 1, 0))

    def test_rejections_surviving_repair_are_not_an_infra_error(self):
        first = {"status": "error", "facts_written": 0, "facts_rejected": 1, "message": "no facts written",
                 "rejected_facts": [{"index": 0, "reason": "r", "fact": {"attributes": {}}}]}
        with patch.object(news_kg, "_write", return_value=first), \
                patch.object(news_kg, "repair_facts", return_value=[]):
            res = news_kg.write_story(self.story(), [{}], RUN_DATE)
        self.assertEqual((res["status"], res["facts_rejected"], res["message"]), ("rejected", 1, None))

    def test_run_fails_only_when_rejections_outnumber_writes(self):
        self.assertIsNone(news_kg.rejection_alert({"facts_written": 38, "facts_rejected": 0}))
        self.assertIsNone(news_kg.rejection_alert({"facts_written": 5, "facts_rejected": 2}))
        self.assertIsNone(news_kg.rejection_alert({}))
        self.assertIn("16 facts rejected", news_kg.rejection_alert({"facts_written": 5, "facts_rejected": 16}))

    def test_tier_counts_distinct_fetched_outlets(self):
        story = self.story()
        self.assertEqual(news_kg.trust_tier(story), "corroborated")
        story["articles"] = story["articles"][:1]
        self.assertEqual(news_kg.trust_tier(story), "single")

    def test_extraction_prompt_keeps_shared_rules(self):
        self.assertIn("generic common noun", news_kg.NEWS_EXTRACTION_SYSTEM)
        self.assertIn('"article": int', news_kg.NEWS_EXTRACTION_SYSTEM)
        self.assertIn("NOT the article's publication date", news_kg.NEWS_EXTRACTION_SYSTEM)
        self.assertIn("from THAT article's publication date", news_kg.NEWS_EXTRACTION_SYSTEM)
        self.assertIn("exact same name in every fact", news_kg.NEWS_EXTRACTION_SYSTEM)
        self.assertIn("South Korea (not 한국, 대한민국)", news_kg.NEWS_EXTRACTION_SYSTEM)


if __name__ == "__main__":
    unittest.main()
