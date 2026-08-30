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


def _glossary_validate_checks() -> None:
    import re as _re

    from runtime_tools.archival_translation.core import RUSSIAN, validate

    terms = [{"ru": "НКВД", "ko": "내무인민위원부", "pattern": _re.compile("НКВД")}]
    chunk = [(1, {"tag": "p", "lines": ["Приказ НКВД о мобилизации."]})]
    bad = {1: ["동원에 관한 인민내무위원부(НКВД) 명령."]}   # 뒤집힌 표기
    good = {1: ["동원에 관한 내무인민위원부(НКВД) 명령."]}
    problems = validate(chunk, bad, RUSSIAN, terms)
    check("glossary violation flagged", any("용어표 미준수" in p and "내무인민위원부" in p for p in problems))
    check("glossary compliance passes", validate(chunk, good, RUSSIAN, terms) == [])
    check("validate without terms unchanged", validate(chunk, good, RUSSIAN) == [])


def _tm_prefill_checks() -> None:
    from runtime_tools.archival_translation import core

    with tempfile.TemporaryDirectory() as td:
        old_db = tm.DEFAULT_DB
        tm.DEFAULT_DB = Path(td) / "tm.sqlite3"
        try:
            tm.record_segments(
                [("Приказ № 1.", "명령 제1호.")], lang_pair="ru-ko",
                doc_id="old-spec", status="published",
            )
            tm.record_segments(
                [("Приказ № 2.", "명령 제2호.")], lang_pair="ru-ko",
                doc_id="old-spec", status="machine",
            )
            docs = [{"offset": 10, "blocks": [
                {"tag": "p", "lines": ["Приказ № 1."]},
                {"tag": "p", "lines": ["Приказ № 2."]},
                {"tag": "p", "lines": []},
            ]}]
            events: list[dict] = []
            filled = core._tm_prefill(docs, core.RUSSIAN, events.append)
            check("prefill reuses published segment", filled.get(10) == ["명령 제1호."])
            check("prefill refuses machine segment", 11 not in filled)
            check("prefill skips empty blocks", 12 not in filled)
            check("prefill emits tmReuse event", events and events[0]["event"] == "tmReuse")
            # 재사용된 블록이 덮는 청크는 실행 대상에서 빠진다
            chunks = [[(10, docs[0]["blocks"][0]), (12, docs[0]["blocks"][2])],
                      [(11, docs[0]["blocks"][1])]]
            runnable = [c for c in chunks
                        if not all((idx in filled) or not b["lines"] for idx, b in c)]
            check("fully covered chunk skipped", runnable == [chunks[1]])
        finally:
            tm.DEFAULT_DB = old_db


def main() -> int:
    _tm_checks()
    _helper_checks()
    _field_checks()
    _post_edit_checks()
    _backfill_align_checks()
    _status_filter_checks()
    _glossary_validate_checks()
    _tm_prefill_checks()
    if FAILURES:
        print(f"{len(FAILURES)} failure(s)")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
