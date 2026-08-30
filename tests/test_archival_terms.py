"""사료 용어 추출의 결정론 부분 — 파싱·집계·보고. 모델 호출은 바꿔 끼운다."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from runtime_tools.archival_translation import terms
from runtime_tools.archival_translation.core import Options, _pattern, _variants


def _glossary():
    # build_glossary와 같은 모양: extra 항목은 표면 그대로, 인명은 곡용 변형.
    return [
        {"ru": "Сокольников", "ko": "소콜니코프", "pattern": _pattern(_variants("Сокольников"))},
        {"ru": "Наркомфин", "ko": "재무인민위원부", "pattern": _pattern(["Наркомфин"])},
        {"ru": "Союз", "ko": "의원 그룹", "pattern": _pattern(["Союз"])},
        {"ru": "Ларин", "ko": "라린", "pattern": _pattern(_variants("Ларин"))},
        {"ru": "Центральный Комитет", "ko": "중앙위원회", "pattern": _pattern(["Центральный Комитет"])},
    ]


def _term(block, surface, lemma=None, kind="person", sense="", target=None, proposed=None,
          glossary=None):
    return {"block": block, "surface": surface, "lemma": lemma or surface, "kind": kind,
            "sense": sense, "target": target, "proposed": proposed, "glossary": glossary}


def _rec(terms_, blocks=(1, 99), offered=(), misfires=()):
    return {"blocks": list(blocks), "offered": list(offered), "misfires": list(misfires),
            "terms": terms_}


class ParseExtraction(unittest.TestCase):
    def test_reads_fenced_json_and_normalises_fields(self):
        raw = ('```json\n{"terms": [{"block": "12", "surface": "Сокольникова", '
               '"lemma": "Сокольников", "kind": "Person", "sense": "재무인민위원", '
               '"glossary": "Сокольников", "target": "소콜니코프"}, '
               '{"block": 13, "surface": "", "lemma": "x"}, '
               '{"block": "n/a", "surface": "y"}], "misfires": ["Союз", ""]}\n```')
        got = terms.parse_extraction(raw)
        self.assertEqual(len(got["terms"]), 1)
        t = got["terms"][0]
        self.assertEqual((t["block"], t["kind"], t["target"], t["glossary"]),
                         (12, "person", "소콜니코프", "Сокольников"))
        self.assertEqual(got["misfires"], ["Союз"])

    def test_target_loses_copied_gloss_and_plural(self):
        for raw, want in [("사르키스(Саркис)", "사르키스"), ("중앙위원회(ЦК)", "중앙위원회"),
                          ("캉성(康生)", "캉성"), ("중농들", "중농"),
                          ("전연방공산당(볼셰비키)", "전연방공산당(볼셰비키)"),
                          ("조직국(Оргбюро ЦК)", "조직국"), ("", None), (None, None)]:
            self.assertEqual(terms._clean_target(raw), want, raw)

    def test_unknown_kind_falls_back_to_term_and_lemma_to_surface(self):
        got = terms.parse_extraction('{"terms": [{"block": 1, "surface": "нэп", "kind": "policy"}]}')
        self.assertEqual(got["terms"][0]["kind"], "term")
        self.assertEqual(got["terms"][0]["lemma"], "нэп")
        self.assertIsNone(got["terms"][0]["glossary"])

    def test_garbage_is_none(self):
        self.assertIsNone(terms.parse_extraction(""))
        self.assertIsNone(terms.parse_extraction("죄송합니다, 추출할 수 없습니다."))
        self.assertIsNone(terms.parse_extraction("{not json"))
        self.assertIsNone(terms.parse_extraction("[1, 2]"))


class MatchGlossary(unittest.TestCase):
    def test_declined_surname_matches_whole(self):
        g = _glossary()
        self.assertEqual(terms.match_glossary("Сокольников", ["Сокольникову"], g)["ko"], "소콜니코프")
        self.assertEqual(terms.match_glossary("", ["Сокольникова"], g)["ko"], "소콜니코프")

    def test_partial_match_is_not_a_match(self):
        g = _glossary()
        self.assertIsNone(terms.match_glossary("Центральный Комитет партии", [], g))
        self.assertIsNone(terms.match_glossary("Советский Союз", ["Советского Союза"], g))
        self.assertEqual(terms.match_glossary("Союз", [], g)["ko"], "의원 그룹")


class Aggregate(unittest.TestCase):
    def test_groups_by_lemma_only_and_keeps_targets_links_per_block(self):
        records = [
            _rec([_term(1, "Сокольникова", "Сокольников", sense="재무인민위원", target="소콜니코프",
                        glossary="Сокольников"),
                  _term(2, "Сокольникову", "Сокольников", sense="재무인민위원", target="소콜니코프")]),
            _rec([_term(3, "Сокольников", "сокольников", kind="org", sense="재무인민위원",
                        target="소콜르니코프", glossary="Сокольников"),
                  _term(3, "Ларин", target=None)]),
        ]
        groups = terms.aggregate(records)
        by = {g["lemma"].lower(): g for g in groups}
        person = by["сокольников"]
        self.assertEqual(person["count"], 3)
        self.assertEqual(person["kind"], "person")  # 최빈 kind
        self.assertEqual(person["targets"], {"소콜니코프": [1, 2], "소콜르니코프": [3]})
        self.assertEqual(person["links"], {"Сокольников": 2})
        self.assertEqual(person["targetLinks"], {"소콜니코프": {"Сокольников": [1]},
                                                 "소콜르니코프": {"Сокольников": [3]}})
        self.assertEqual(person["surfaces"], ["Сокольникова", "Сокольникову", "Сокольников"])
        self.assertEqual(by["ларин"]["missing"], [3])
        self.assertEqual(groups[0]["lemma"], "Сокольников")  # 빈도순


class PreReport(unittest.TestCase):
    def test_candidates_exclude_registered_and_rare_and_collect_misfires(self):
        records = [
            _rec([_term(1, "Сокольникова", "Сокольников", sense="재무인민위원", glossary="Сокольников"),
                  _term(1, "Богушевского", "Богушевский", sense="레닌그라드 대의원", proposed="보구솁스키"),
                  _term(2, "Богушевский", "Богушевский", sense="레닌그라드 대의원", proposed="보구셰프스키"),
                  _term(2, "Шанин", proposed="샤닌"),
                  _term(2, "Ларину", "Ларин", proposed="라린")],  # LLM 연결 없어도 표면 일치로 등재
                 offered=["Союз", "Наркомфин", "Сокольников"], misfires=["Союз", "Ленин"]),
            _rec([_term(3, "Богушевскому", "Богушевский", proposed="보구솁스키")],
                 blocks=(3, 4), offered=["Союз"]),
        ]
        sources = {1: "Товарищ Сокольникова и Богушевского.", 2: "Богушевский сказал. Шанин. Ларину.",
                   3: "Богушевскому ответили."}
        rep = terms.pre_report(records, _glossary(), sources, min_count=2)
        self.assertEqual([c["lemma"] for c in rep["candidates"]], ["Богушевский"])
        c = rep["candidates"][0]
        self.assertEqual(c["proposed"], "보구솁스키")  # 다수 제안
        self.assertEqual(c["proposedAll"], ["보구솁스키", "보구셰프스키"])
        self.assertIn("Богушевского", c["context"])
        self.assertEqual(rep["extraSnippet"], {"Богушевский": "보구솁스키"})
        self.assertEqual(sorted(r["lemma"] for r in rep["registered"]), ["Ларин", "Сокольников"])
        # 오탐: Союз는 2청크 제시 중 1청크 오탐 → exclude 후보 아님. Ленин은 제시 목록에 없어 무시.
        self.assertEqual(rep["misfires"], [{"ru": "Союз", "offered": 2, "misfired": 1, "always": False}])
        self.assertEqual(rep["excludeSnippet"], [])

    def test_always_misfired_entry_is_exclude_candidate(self):
        records = [_rec([], blocks=(1, 1), offered=["Союз"], misfires=["Союз"]),
                   _rec([], blocks=(2, 2), offered=["Союз"], misfires=["Союз"])]
        rep = terms.pre_report(records, _glossary(), {}, min_count=1)
        self.assertEqual(rep["excludeSnippet"], ["Союз"])


class PostReport(unittest.TestCase):
    def setUp(self):
        self.spec = {"postEdits": {"카메네바에게": "카메네프에게", "재정인민위원부": "재무인민위원부",
                                   "소비에트 소유즈": "소비에트 연방", "루리예)": "라린"}}
        self.targets = {
            1: ["카메네프가 말했습니다."], 2: ["카메네바에게 답했습니다."], 3: ["카메네바는 웃었습니다."],
            4: ["재정인민위원부의 안입니다."], 5: ["소련 노동조합이 그렇습니다."],
            6: ["신반대파의 결의입니다."], 7: ["새 반대파의 결의입니다."],
            8: ["소비에트 소유즈가 있습니다."], 9: ["소비에트 소유즈는 큽니다."], 10: ["소련이 있습니다."],
            11: ["라린(루리예)이 말했습니다."], 12: ["루리예가 말했습니다."], 13: ["라린이 말했습니다."],
            14: ["연방을 이룹니다."],
        }

    def _records(self):
        return [_rec([
            _term(1, "Каменев", sense="정치국원", target="카메네프"),
            _term(2, "Каменеву", "Каменев", sense="정치국원", target="카메네바"),
            _term(3, "Каменева", "Каменев", sense="정치국원", target="카메네바"),
            _term(4, "Наркомфин", kind="org", sense="재무 담당 인민위원부", target="재정인민위원부",
                  glossary="Наркомфин"),
            # 용어표 Союз(의원 그룹)와 표면이 겹치지만 LLM은 연결하지 않았다 → 이탈 아님
            _term(5, "Союз", kind="org", sense="소련", target="소련"),
            _term(14, "Союз", kind="org", sense="소련", target="연방"),
            _term(6, "новая оппозиция", kind="term", sense="지노비예프 그룹", target="신반대파"),
            _term(7, "новая оппозиция", kind="term", sense="지노비예프 그룹", target="새 반대파"),
            # 다수 표기가 이미 postEdits로 고쳐진 오역 → 기준은 남은 쪽
            _term(8, "Советский Союз", kind="place", sense="나라", target="소비에트 소유즈"),
            _term(9, "Советский Союз", kind="place", sense="나라", target="소비에트 소유즈"),
            _term(10, "Советский Союз", kind="place", sense="나라", target="소련"),
            # 용어표 라린: 루리예 ×2 중 11은 postEdits가 덮고 12는 아니다
            _term(11, "Ларин", sense="경제학자", target="루리예", glossary="Ларин"),
            _term(12, "Ларина", "Ларин", sense="경제학자", target="루리예", glossary="Ларин"),
            _term(13, "Ларин", sense="경제학자", target="라린", glossary="Ларин"),
        ], offered=["Наркомфин", "Союз", "Ларин"])]

    def test_glossary_rendering_is_canonical_and_suggestions_do_not_contradict(self):
        rep = terms.post_report(self._records(), _glossary(), self.spec, self.targets)
        by = {i["lemma"]: i for i in rep["inconsistent"]}
        larin = by["Ларин"]
        self.assertEqual(larin["canonical"], "라린")
        self.assertEqual(larin["majority"], "루리예")  # 남은 블록 수 기준 표시는 그대로
        self.assertEqual(rep["postEditsSnippet"].get("루리예"), "라린")
        self.assertNotIn("라린", rep["postEditsSnippet"])

    def test_covered_majority_is_not_canonical(self):
        rep = terms.post_report(self._records(), _glossary(), self.spec, self.targets)
        su = {i["lemma"]: i for i in rep["inconsistent"]}["Советский Союз"]
        self.assertEqual(su["canonical"], "소련")
        self.assertEqual(su["majority"], "소련")  # 남은 블록이 있는 표기가 앞에 온다
        self.assertEqual(su["variants"][0]["target"], "소비에트 소유즈")
        self.assertEqual(su["variants"][0]["covered"], [8, 9])
        self.assertNotIn("소비에트 연방", rep["postEditsSnippet"])
        self.assertNotIn("소련", rep["postEditsSnippet"])

    def test_inconsistency_without_glossary_uses_first_remaining_majority(self):
        rep = terms.post_report(self._records(), _glossary(), self.spec, self.targets)
        by = {i["lemma"]: i for i in rep["inconsistent"]}
        kam = by["Каменев"]
        self.assertEqual(kam["canonical"], "카메네프")  # 카메네바 ×2 중 하나는 postEdits가 덮음 → 1:1, 첫 등장
        v = kam["variants"][0]
        self.assertEqual((v["target"], v["covered"], v["remaining"]), ("카메네바", [2], [3]))
        self.assertEqual(rep["postEditsSnippet"]["카메네바"], "카메네프")
        self.assertEqual(rep["postEditsSnippet"]["새 반대파"], "신반대파")

    def test_deviation_only_for_llm_linked_occurrences(self):
        rep = terms.post_report(self._records(), _glossary(), self.spec, self.targets)
        dev = {(d["lemma"], d["target"]): d for d in rep["deviations"]}
        d = dev[("Наркомфин", "재정인민위원부")]
        self.assertEqual((d["covered"], d["remaining"]), ([4], []))
        self.assertNotIn("재정인민위원부", rep["postEditsSnippet"])
        self.assertNotIn(("Союз", "소련"), dev)
        self.assertNotIn(("Союз", "연방"), dev)
        # 연결이 없으니 용어표 표기('소유즈 (의원 그룹)')로 끌려가지 않는다; 같은 sense의
        # 두 표기(소련/연방)는 용어표와 무관한 불일치로서 기준 표기 쪽으로 제안된다.
        self.assertEqual(rep["postEditsSnippet"].get("연방"), "소련")
        self.assertEqual(dev[("Ларин", "루리예")]["remaining"], [12])

    def test_sense_disjoint_variant_without_glossary_is_not_suggested(self):
        records = [_rec([
            _term(1, "совет", kind="term", sense="소비에트 체제", target="소비에트"),
            _term(2, "совет", kind="term", sense="조언", target="충고"),
        ])]
        rep = terms.post_report(records, [], {}, {1: ["소비에트"], 2: ["충고"]})
        item = rep["inconsistent"][0]
        self.assertTrue(item["variants"][0]["senseDisjoint"])
        self.assertEqual(rep["postEditsSnippet"], {})

    def test_unlinked_variant_under_glossary_entry_is_not_suggested(self):
        records = [_rec([
            _term(1, "Союз", kind="org", sense="의원 그룹", target="의원 그룹", glossary="Союз"),
            _term(2, "Союз", kind="org", sense="소련", target="연방"),
        ], offered=["Союз"])]
        rep = terms.post_report(records, _glossary(), {}, {1: ["의원 그룹"], 2: ["연방"]})
        item = rep["inconsistent"][0]
        self.assertEqual(item["canonical"], "의원 그룹")
        self.assertFalse(item["variants"][0]["linked"])
        self.assertEqual(rep["postEditsSnippet"], {})
        self.assertEqual(rep["deviations"], [])

    def test_span_mismatch_is_reported_but_not_suggested(self):
        gl = [{"ru": "стабилизация капитализма", "ko": "자본주의의 안정화",
               "pattern": _pattern(["стабилизация капитализма"])},
              {"ru": "РКП(б)", "ko": "러시아공산당(볼셰비키)", "pattern": _pattern(["РКП(б)"])}]
        records = [_rec([
            _term(1, "стабилизации капитализма", "стабилизация капитализма", kind="term",
                  target="안정화", glossary="стабилизация капитализма"),
            _term(2, "ЦК РКП(б)", kind="org", target="러시아공산당(볼셰비키) 중앙위원회", glossary="РКП(б)"),
            _term(3, "ЦК РКП(б)", kind="org", target="러시아공산당(볼셰비키)", glossary="РКП(б)"),
        ], offered=["стабилизация капитализма", "РКП(б)"])]
        targets = {1: ["안정화"], 2: ["러시아공산당(볼셰비키) 중앙위원회"], 3: ["러시아공산당(볼셰비키)"]}
        rep = terms.post_report(records, gl, {}, targets)
        self.assertEqual(sorted(d["target"] for d in rep["deviations"] if d["spanMismatch"]),
                         sorted(["안정화", "러시아공산당(볼셰비키) 중앙위원회"]))
        self.assertEqual(rep["postEditsSnippet"], {})
        rkp = next(i for i in rep["inconsistent"] if i["lemma"] == "ЦК РКП(б)")
        # 남은 블록 1:1이라 첫 등장(긴 표기)이 '다수'로 표시되지만 기준은 용어표 표기다
        self.assertEqual((rkp["majority"], rkp["canonical"]),
                         ("러시아공산당(볼셰비키) 중앙위원회", "러시아공산당(볼셰비키)"))
        self.assertEqual(rkp["variants"][0]["target"], "러시아공산당(볼셰비키)")
        md = terms.render_post_markdown({"id": "t"}, rep, [])
        self.assertIn("범위 차이", md)

    def test_markdown_renders_every_section(self):
        rep = terms.post_report(self._records(), _glossary(), self.spec, self.targets)
        md = terms.render_post_markdown({"id": "t"}, rep, [9])
        for needle in ("표기 불일치", "용어표 이탈", "postEdits 제안", "Каменев", "Наркомфин",
                       "postEdits 적용됨", "기준: **라린**", "번역을 찾지 못한 원문 블록 1개"):
            self.assertIn(needle, md)


class CachedTargets(unittest.TestCase):
    def test_latest_record_wins_and_empty_blocks_ignored(self):
        lines = [json.dumps({"key": "a", "blocks": {"4": ["옛"], "5": []}}),
                 json.dumps({"key": "b", "blocks": {"4": ["새"]}}), ""]
        self.assertEqual(terms.cached_targets(lines), {4: ["새"]})

    def test_align_applies_post_edits(self):
        lines = [json.dumps({"key": "a", "blocks": {"4": ["재정인민위원부"], "9": ["고아"]}})]
        pairs, ids = terms.align_cached_blocks({4: "Наркомфин"}, lines,
                                               {"postEdits": {"재정": "재무"}})
        self.assertEqual((pairs, ids), ([("Наркомфин", "재무인민위원부")], [4]))


class Extract(unittest.TestCase):
    def _chunk(self):
        return [(10, {"tag": "p", "lines": ["Сокольников выступал."]}),
                (11, {"tag": "p", "lines": []})]

    def test_retries_once_then_caches_success_and_drops_unoffered_links(self):
        calls = []

        def fake_call(prompt):
            calls.append(prompt)
            if len(calls) == 1:
                return "no json here"
            return json.dumps({"terms": [
                {"block": 10, "surface": "Сокольников", "kind": "person", "glossary": "Сокольников"},
                {"block": 10, "surface": "Ленин", "kind": "person", "glossary": "Ленин"},
            ], "misfires": []})

        events = []
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(terms, "_call", fake_call), \
                mock.patch.object(terms, "_chunk_key", lambda mode, prompt: f"{mode}:{hash(prompt)}"), \
                mock.patch.object(terms.time, "sleep", lambda *_: None):
            cache = terms._JsonlCache(Path(d) / "t.jsonl")
            recs = terms.extract([self._chunk()], "pre", _glossary(), Options(), cache,
                                 emit=events.append, concurrency=1)
            self.assertEqual(len(calls), 2)
            self.assertIsNone(recs[0]["error"])
            self.assertEqual(recs[0]["offered"], ["Сокольников"])
            self.assertEqual([t["glossary"] for t in recs[0]["terms"]], ["Сокольников", None])
            self.assertFalse(recs[0]["cached"])
            self.assertEqual([e["event"] for e in events], ["extractRetry", "chunk"])
            # 두 번째 실행은 캐시 적중 — 호출 없음
            cache2 = terms._JsonlCache(Path(d) / "t.jsonl")
            recs2 = terms.extract([self._chunk()], "pre", _glossary(), Options(), cache2,
                                  concurrency=1)
            self.assertEqual(len(calls), 2)
            self.assertTrue(recs2[0]["cached"])
            self.assertEqual([t["glossary"] for t in recs2[0]["terms"]], ["Сокольников", None])

    def test_two_failures_yield_error_record_not_exception(self):
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(terms, "_call", lambda prompt: None), \
                mock.patch.object(terms, "_chunk_key", lambda mode, prompt: "k"), \
                mock.patch.object(terms.time, "sleep", lambda *_: None):
            cache = terms._JsonlCache(Path(d) / "t.jsonl")
            recs = terms.extract([self._chunk()], "post", [], Options(), cache,
                                 targets={10: ["소콜니코프가 발언했습니다."]}, concurrency=1)
            self.assertEqual(recs[0]["error"], "빈 응답")
            self.assertEqual(recs[0]["terms"], [])
            self.assertEqual(cache.data, {})

    def test_post_chunk_without_any_target_is_skipped_without_call(self):
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(terms, "_call", side_effect=AssertionError("must not call")):
            cache = terms._JsonlCache(Path(d) / "t.jsonl")
            recs = terms.extract([self._chunk()], "post", [], Options(), cache, targets={},
                                 concurrency=1)
            self.assertEqual(recs[0]["terms"], [])
            self.assertTrue(recs[0]["cached"])

    def test_prompts_carry_offered_glossary_and_pairs(self):
        offered = [{"ru": "Союз", "ko": "의원 그룹"}]
        pre = terms.render_pre_prompt(self._chunk(), offered)
        self.assertIn("- Союз → 의원 그룹", pre)
        self.assertIn("[[10|p]]\nСокольников выступал.", pre)
        self.assertNotIn("[[11|p]]", pre)  # 빈 블록은 보내지 않는다
        post = terms.render_post_prompt(self._chunk(), offered, {10: ["소콜니코프가 발언했습니다."]})
        self.assertIn("원문: Сокольников выступал.\n번역: 소콜니코프가 발언했습니다.", post)


if __name__ == "__main__":
    unittest.main()
