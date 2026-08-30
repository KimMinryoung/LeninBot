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
            [("товарищ", "동지")], lang_pair="ru-ko", doc_id="d2", status="reviewed", db_path=db
        )
        got = tm.exact_matches(["товарищ", "нет такого", ""], lang_pair="ru-ko", db_path=db)
        check("reviewed row wins lookup", got.get("товарищ") == "동지")
        check("missing source absent from lookup", "нет такого" not in got)
        check("other lang pair is invisible", tm.exact_matches(["товарищ"], lang_pair="zh-ko", db_path=db) == {})

        s = tm.stats(db_path=db)
        check("stats totals", s["total"] == 4 and s["byLangPair"].get("ru-ko") == 4)
        check("stats statuses", s["byStatus"].get("machine") == 3 and s["byStatus"].get("reviewed") == 1)


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


def main() -> int:
    _tm_checks()
    _helper_checks()
    _field_checks()
    if FAILURES:
        print(f"{len(FAILURES)} failure(s)")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
