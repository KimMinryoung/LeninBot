#!/usr/bin/env python3
"""Offline smoke for the translation memory and the shared translation checks.

No network, no DB server, no frontend checkout — unlike
smoke_archival_translation.py this suite must stay runnable in a bare clone,
because the TM and the deterministic validators are exactly the parts that
should be testable without credentials. run_smokes.sh picks it up via the
scripts/smoke_*.py glob.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime_tools import translation_memory as tm
from scripts._translation_common import (
    field_translation_problems,
    hangul_ratio,
    parse_json_object,
    strip_code_fences,
    tag_sequence,
    url_multiset,
)

FAILURES: list[str] = []


def check(name: str, cond: bool) -> None:
    if cond:
        print(f"ok   {name}")
    else:
        FAILURES.append(name)
        print(f"FAIL {name}")


def _tm_checks() -> None:
    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "tm.sqlite3"
        first = [("Приказ № 00447", "명령 제00447호"), ("", "무시된다"), ("원문만", "")]
        inserted = tm.record_segments(
            first, lang_pair="ru-ko", doc_id="spec-a", block_ids=[1, 2, 3], db_path=db
        )
        check("record skips empty sides", inserted == 1)
        again = tm.record_segments(
            first, lang_pair="ru-ko", doc_id="spec-a", block_ids=[1, 2, 3], db_path=db
        )
        check("record is idempotent", again == 0)
        other_doc = tm.record_segments(
            [("Приказ № 00447", "명령 제00447호")], lang_pair="ru-ko", doc_id="spec-b", db_path=db
        )
        check("same pair from another doc is kept", other_doc == 1)

        tm.record_segments([("товарищ", "동무")], lang_pair="ru-ko", doc_id="d1", db_path=db)
        tm.record_segments(
            [("товарищ", "벗")], lang_pair="ru-ko", doc_id="d3", status="published", db_path=db
        )
        got = tm.exact_matches(["товарищ"], lang_pair="ru-ko", db_path=db)
        check("published row beats machine", got.get("товарищ") == "벗")
        tm.record_segments(
            [("товарищ", "동지")], lang_pair="ru-ko", doc_id="d2", status="reviewed", db_path=db
        )
        got = tm.exact_matches(["товарищ", "нет такого", ""], lang_pair="ru-ko", db_path=db)
        check("reviewed row beats published", got.get("товарищ") == "동지")
        check("missing source absent from lookup", "нет такого" not in got)
        check("other lang pair is invisible", tm.exact_matches(["товарищ"], lang_pair="zh-ko", db_path=db) == {})

        # 같은 쌍이 나중에 더 높은 상태로 오면 승격되고, 낮은 상태로는 내려가지 않는다
        upgraded = tm.record_segments(
            [("товарищ", "동무")], lang_pair="ru-ko", doc_id="d1", status="published", db_path=db
        )
        check("machine row upgraded to published", upgraded == 1)
        downgraded = tm.record_segments(
            [("товарищ", "동무")], lang_pair="ru-ko", doc_id="d1", status="machine", db_path=db
        )
        check("status never downgrades", downgraded == 0)

        s = tm.stats(db_path=db)
        check("stats totals", s["total"] == 5 and s["byLangPair"].get("ru-ko") == 5)
        check(
            "stats statuses",
            s["byStatus"].get("machine") == 2
            and s["byStatus"].get("published") == 2
            and s["byStatus"].get("reviewed") == 1,
        )


def _helper_checks() -> None:
    check("strip fences", strip_code_fences('```json\n{"a": 1}\n```') == '{"a": 1}')
    check("strip fences leaves bare text", strip_code_fences("plain") == "plain")
    check("json fallback slice", parse_json_object('Here is the JSON: {"a": 1} thanks')["a"] == 1)
    check("tag sequence normalizes case and closers", tag_sequence("<P>x</p><BR>") == ["p", "p", "br"])
    check("tag sequence ignores comments", tag_sequence("<!-- <div> --><p>x</p>") == ["p", "p"])
    check(
        "url multiset trims trailing punctuation",
        url_multiset("see https://a.example/x.") == {"https://a.example/x": 1},
    )
    # "한국 economy 분석" = 한글 4자 / 알파벳 계열 11자
    check("hangul ratio on mixed text", abs(hangul_ratio("한국 economy 분석") - 4 / 11) < 1e-9)


def _field_checks() -> None:
    src = '<p>한국 경제에 대한 분석. <a href="https://ex.com/r">링크</a></p>' * 5
    ok_target = '<p>An analysis of the Korean economy. <a href="https://ex.com/r">link</a></p>' * 5
    check("clean field passes", field_translation_problems(src, ok_target, label="content_en") == [])

    echo = field_translation_problems("한국 경제 회고", "한국 경제 회고", label="title_en")
    check("verbatim echo flagged", any("untranslated" in p for p in echo))

    short_ok = field_translation_problems("한국 경제 회고", "A Korean Economy Retrospective (한국)", label="title_en")
    check("short title may quote Korean", short_ok == [])

    hangul_left = field_translation_problems(
        src,
        '<p>한국 경제에 대한 분석이다. <a href="https://ex.com/r">링크</a></p>' * 5,
        label="content_en",
    )
    check("residual hangul flagged", any("Hangul" in p for p in hangul_left))

    tag_broken = field_translation_problems(
        src, ok_target.replace("<a ", "<span ", 1), label="content_en"
    )
    check("broken tag sequence flagged", any("tag sequence" in p for p in tag_broken))

    url_broken = field_translation_problems(
        src, ok_target.replace("https://ex.com/r", "https://ex.com/en/r", 1), label="content_en"
    )
    check("changed URL flagged", any("URLs differ" in p for p in url_broken))

    check(
        "non-korean source is exempt from hangul checks",
        field_translation_problems("Already English text here", "Already English text here", label="x") == [],
    )


def _post_edit_checks() -> None:
    from runtime_tools.archival_translation.core import apply_post_edits

    spec = {"postEdits": {"인민내무위원부": "내무인민위원부", "Ульмером": "울메르와"}}
    lines = ["인민내무위원부(НКВД)의 명령.", "Ульмером 함께.", "무관한 줄."]
    fixed = apply_post_edits(lines, spec)
    check(
        "post edits substituted",
        fixed == ["내무인민위원부(НКВД)의 명령.", "울메르와 함께.", "무관한 줄."],
    )
    check("empty post edits pass through", apply_post_edits(lines, {}) == lines)
    check("line count preserved", len(fixed) == len(lines))


def _backfill_align_checks() -> None:
    import json as _json

    from scripts.backfill_translation_memory import align_cached_blocks

    source = {5: "Приказ № 1.", 6: "Приказ № 2.", 9: "Подпись."}
    cache_lines = [
        _json.dumps({"key": "old", "blocks": {"5": ["옛 번역."], "7": ["원문에 없는 블록."]}}),
        "",
        _json.dumps({"key": "new", "blocks": {"5": ["명령 제1호."], "6": ["명령 제2호."], "8": []}}),
    ]
    spec = {"postEdits": {"명령 제2호": "제2호 명령"}}
    pairs, block_ids = align_cached_blocks(source, cache_lines, spec)
    check("later cache record wins", ("Приказ № 1.", "명령 제1호.") in pairs)
    check("stale target for same block dropped", all(t != "옛 번역." for _, t in pairs))
    check("block absent from source skipped", 7 not in block_ids)
    check("empty cached block skipped", 8 not in block_ids)
    check("post edits applied in alignment", ("Приказ № 2.", "제2호 명령.") in pairs)
    check("block ids sorted and aligned", block_ids == [5, 6])


def _status_filter_checks() -> None:
    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "tm.sqlite3"
        tm.record_segments([("приказ", "명령")], lang_pair="ru-ko", doc_id="m", db_path=db)
        tm.record_segments(
            [("подпись", "서명")], lang_pair="ru-ko", doc_id="p", status="published", db_path=db
        )
        got = tm.exact_matches(
            ["приказ", "подпись"], lang_pair="ru-ko",
            statuses=("published", "reviewed"), db_path=db,
        )
        check("status filter hides machine rows", "приказ" not in got)
        check("status filter keeps published rows", got.get("подпись") == "서명")


def _validate_checks() -> None:
    """검증기는 형식·누락·미번역·길이만 본다. 용어표 표기는 검사하지 않는다."""
    import re as _re

    from llm import call_registry
    from runtime_tools.archival_translation import core
    from runtime_tools.archival_translation.core import RUSSIAN, validate

    chunk = [(1, {"tag": "p", "lines": ["Приказ НКВД о мобилизации."]})]
    check("glossary rendering is not validated",
          validate(chunk, {1: ["동원에 관한 인민내무위원부(НКВД) 명령."]}, RUSSIAN) == [])
    check("verbatim echo still fails",
          any("그대로 반환" in p for p in validate(chunk, {1: ["Приказ НКВД о мобилизации."]}, RUSSIAN)))

    glossary = [{"ru": "НКВД", "ko": "내무인민위원부", "pattern": _re.compile("НКВД")}]

    class _StubCache:
        def get(self, key):
            return None

        def put(self, key, blocks, meta):
            pass

    original = call_registry.generate_sync
    calls: list[int] = []
    try:
        # 용어표와 다른 표기라도 형식이 맞으면 1회에 통과
        call_registry.generate_sync = lambda *a, **k: (calls.append(1) or "[[1|p]]\n동원에 관한 인민내무위원부(НКВД) 명령.")
        stats = core.Stats()
        got = core._translate_chunk(chunk, glossary, _StubCache(), core.Options(retries=2), stats, lambda e: None)
        check("different rendering accepted first try", len(calls) == 1 and stats.translated == 1 and 1 in got)
        # 치명 문제(원문 그대로 반환)는 여전히 실패로 올라온다
        call_registry.generate_sync = lambda *a, **k: "[[1|p]]\nПриказ НКВД о мобилизации."
        try:
            core._translate_chunk(chunk, glossary, _StubCache(), core.Options(retries=1), core.Stats(), lambda e: None)
            check("fatal problems still fail", False)
        except RuntimeError:
            check("fatal problems still fail", True)
    finally:
        call_registry.generate_sync = original


def _prepare_scan_checks() -> None:
    import re as _re

    from scripts.scan_archival_terms import scan_candidates

    docs = [{"offset": 0, "blocks": [{"lines": [
        "Приказ НКВД получен.",
        "Доклад товарища Ульмера. Приказ прилагается.",
        "Сообщение Ульмера в ЦК.",
    ]}]}]
    glossary = [{"ru": "НКВД", "ko": "내무인민위원부",
                 "pattern": _re.compile(r"(?<![А-Яа-яЁё])НКВД(?![А-Яа-яЁё])")}]

    found = scan_candidates(docs, glossary, min_count=1)
    surfaces = {c["surface"]: c for c in found}
    check("known glossary surface excluded", "НКВД" not in surfaces)
    check("mid-sentence name is a candidate", surfaces.get("Ульмера", {}).get("count") == 2)
    check("abbreviation is a candidate", "ЦК" in surfaces)
    check("sentence-initial capitals ignored", "Приказ" not in surfaces and "Доклад" not in surfaces)
    check("context captured", "товарища Ульмера" in surfaces["Ульмера"]["context"])
    check("min_count filters singletons",
          {c["surface"] for c in scan_candidates(docs, glossary, min_count=2)} == {"Ульмера"})


def _tm_example_checks() -> None:
    from runtime_tools.archival_translation import core
    from scripts.suggest_tm_examples import rank_examples

    # 스펙에 고정된 예시가 청크 프롬프트에 실린다
    block = {"tag": "p", "lines": ["Приказ о мобилизации."],
             "tmExamples": [{"source": "Приказ № 00447.", "target": "명령 제00447호."}]}
    prompt = core._chunk_prompt([(1, block)], [], core.Options())
    check("tm examples rendered in prompt",
          "참고 번역례" in prompt and "명령 제00447호." in prompt)
    bare = {"tag": "p", "lines": ["Приказ о мобилизации."]}
    check("no examples, no section",
          "참고 번역례" not in core._chunk_prompt([(1, bare)], [], core.Options()))

    # 후보 순위: 겹침 높은 세그먼트가 앞서고, 완전 일치·범위 밖 길이는 빠진다
    blocks = ["Приказ народного комиссара внутренних дел о мобилизации резервов."]
    segments = [
        ("Приказ народного комиссара внутренних дел об учете резервов.", "내무인민위원 명령.", "published"),
        ("Совершенно другая тема без общих слов тут вообще нигде.", "다른 주제.", "published"),
        ("Приказ народного комиссара внутренних дел о мобилизации резервов.", "완전 일치라 제외.", "reviewed"),
        ("Приказ.", "너무 짧음.", "published"),
    ]
    ranked = rank_examples(blocks, segments, "ru", limit=3, min_score=0.25)
    check("high-overlap segment ranked", ranked and "учете" in ranked[0]["source"])
    check("exact match excluded from examples", all("완전 일치라 제외." != c["target"] for c in ranked))
    check("low-overlap and short filtered", len(ranked) == 1)

    zh = rank_examples(["中央委员会关于修正主义的决定与通知全文如下所述内容。"],
                       [("中央委员会关于修正主义的决定与通知的全文内容如下所示等等。", "중앙위원회 결정.", "published")],
                       "zh", limit=3, min_score=0.2)
    check("zh bigram path works", len(zh) == 1)


def main() -> int:
    _tm_checks()
    _helper_checks()
    _field_checks()
    _post_edit_checks()
    _backfill_align_checks()
    _status_filter_checks()
    _validate_checks()
    _tm_prefill_checks()
    _prepare_scan_checks()
    _tm_example_checks()
    if FAILURES:
        print(f"{len(FAILURES)} failure(s)")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
