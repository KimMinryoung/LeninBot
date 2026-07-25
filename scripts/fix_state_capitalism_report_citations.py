#!/usr/bin/env python3
"""Correct three citation errors in the public report 'state-capitalism-theory-review'.

Found while sourcing the CommuLingo glossary entry from this report. Each item
below was checked against the primary text on marxists.org rather than against
the report itself.

1. The blockquote in 1.A ("자본주의 국가에서 국가자본주의와 프롤레타리아 국가에서
   국가자본주의는 두 개의 서로 다른 개념이다...") could not be located. The prose
   attributes it to 『현물세』(April 1921); footnote [^1] attributes it to the
   Third Congress speech (June 1921); the wording appears in neither, nor in six
   other candidate texts (1921 jul/01, jul/05, mar/15, 10thcong ch03, 1922
   mar/27, 1918 apr/29, 1918 may/09). It is replaced with the passage from the
   same work that carries the identical argument and is verifiable, and the
   footnote is repointed with a URL.

2. Ted Grant's reply to Cliff is dated 1948 in 3.3 and in [^27]. The Marxists
   Internet Archive files it under 1949 as "Against the Theory of State
   Capitalism: Reply to Comrade Cliff".

3. Cliff's publication history in 1.C is compressed to "1948/1955". His own 1988
   introduction gives it precisely, and the familiar title is later than both.

Footnote [^2] was checked too and is correct: the line about state capitalism
being a step forward is a self-quotation from May 1918 that Lenin reproduces
inside 『현물세』, so it does belong to the cited work. It is left alone.

Usage:
  bash scripts/run_fix_state_capitalism_report.sh            # dry run
  bash scripts/run_fix_state_capitalism_report.sh --apply
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db import execute as db_execute, query as db_query
from scripts.fix_broken_report_links import clear_frontend_research_cache

SLUG = "state-capitalism-theory-review"

OLD_QUOTE = (
    '> "자본주의 국가에서 국가자본주의와 프롤레타리아 국가에서 국가자본주의는 두 개의 서로 다른 '
    '개념이다. 자본주의 국가에서 국가자본주의란 국가가 인정하고 국가가 통제하는 자본주의로서 '
    '부르주아지에 유리하고 프롤레타리아트에 반대하는 것을 뜻한다. 프롤레타리아 국가에서 이것은 '
    '노동계급에 유리하게 작동하며, 여전히 강한 부르주아지에 저항하고 그에 맞서 투쟁하기 위한 '
    '목적을 갖는다."[^1]'
)

NEW_QUOTE = (
    '> "융커-자본가 국가, 지주-자본가 국가 대신에 혁명적-민주주의 국가, 곧 모든 특권을 혁명적으로 '
    '폐지하고 가장 완전한 민주주의를 혁명적으로 도입하기를 두려워하지 않는 국가를 놓아 보라. '
    '진정으로 혁명적-민주주의적인 국가라면 국가독점자본주의는 불가피하게 사회주의를 향한 한 걸음, '
    '그것도 한 걸음 이상을 의미하게 된다."[^1]\n'
    '\n'
    '같은 글에서 레닌은 독일을 예로 들어 같은 논점을 되풀이한다. 대규모 계획적 생산조직에서 '
    '"군국주의적·융커적·부르주아적·제국주의적" 국가를 지우고 그 자리에 "사회적 유형과 계급 내용이 '
    '다른 국가, 곧 소비에트 국가, 다시 말해 프롤레타리아 국가"를 놓으면 "사회주의에 필요한 조건의 '
    '총계"가 된다는 것이다. 국가자본주의라는 경제 형태 그 자체가 아니라 그 형태를 쥔 국가의 '
    '계급성이 사회적 성격을 규정한다.'
)

OLD_FOOTNOTE_1 = (
    '[^1]: Lenin, "Third Congress of the Communist International: Speech on the Tax in Kind," '
    'June 1921. 『레닌전집』 32권. Marxists Internet Archive.'
)

NEW_FOOTNOTE_1 = (
    '[^1]: Lenin, "The Tax in Kind," April 1921. 『레닌전집』 32권. '
    'https://www.marxists.org/archive/lenin/works/1921/apr/21.htm — 레닌이 1917년 '
    '『임박한 파국과 그 대책』에서 세운 정식을 이 글에서 다시 인용한 대목이다.'
)

OLD_GRANT_BODY = "Ted Grant의 1948년 반박문은 트로츠키주의 내부에서"
NEW_GRANT_BODY = "Ted Grant의 1949년 반박문은 트로츠키주의 내부에서"

OLD_FOOTNOTE_27 = (
    '[^27]: Ted Grant, "Against the Theory of State Capitalism," 1948. marxist.com.'
)
NEW_FOOTNOTE_27 = (
    '[^27]: Ted Grant, "Against the Theory of State Capitalism: Reply to Comrade Cliff," 1949. '
    'https://www.marxists.org/archive/grant/1949/cliff.htm'
)

OLD_CLIFF_CONTEXT = (
    "냉전 반공 히스테리의 압력 속에서 소련을 자본주의 국가로 재규정했다."
)
NEW_CLIFF_CONTEXT = (
    "냉전 반공 히스테리의 압력 속에서 소련을 자본주의 국가로 재규정했다. 원고는 1947년에 쓰여 "
    "1948년 6월 『스탈린주의 러시아의 성격(The Nature of Stalinist Russia)』이라는 제목으로 등사 "
    "배포되었고, 1955년 개정판이 『스탈린주의 러시아: 마르크스주의적 분석』으로 나왔으며, 오늘날 "
    "통용되는 제목 『러시아의 국가자본주의』는 1974년 플루토판에서 붙은 것이다."
)

CORRECTION_NOTE = (
    "\n---\n\n"
    "*정정 (2026-07-25): 1.A의 인용문은 본문이 『현물세』(1921년 4월), 각주 [^1]이 코민테른 3차 "
    "대회 연설(1921년 6월)로 서로 다르게 출처를 밝혔으나 어느 쪽에서도 해당 표현을 확인할 수 "
    "없어, 같은 논점을 담은 『현물세』의 확인된 구절로 교체하고 각주를 바로잡았다. 3.3과 [^27]의 "
    "Ted Grant 반박문 연도를 1948년에서 1949년으로 정정했다. 1.C에 클리프 저작의 정확한 간행 "
    "이력을 보충했다. [^2]는 재확인 결과 정확하여 그대로 두었다.*\n"
)

REPLACEMENTS = [
    ("1.A 인용문", OLD_QUOTE, NEW_QUOTE),
    ("각주 [^1]", OLD_FOOTNOTE_1, NEW_FOOTNOTE_1),
    ("3.3 Grant 연도", OLD_GRANT_BODY, NEW_GRANT_BODY),
    ("각주 [^27]", OLD_FOOTNOTE_27, NEW_FOOTNOTE_27),
    ("1.C 간행 이력", OLD_CLIFF_CONTEXT, NEW_CLIFF_CONTEXT),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    rows = db_query(
        "SELECT id, slug, status, markdown FROM research_documents WHERE slug = %s",
        (SLUG,),
    )
    if not rows:
        print(f"no document with slug {SLUG}", file=sys.stderr)
        return 1

    row = rows[0]
    text = row["markdown"]
    print(f"{row['slug']} (id={row['id']}, status={row['status']}, {len(text)} chars)")

    failed = False
    for label, old, new in REPLACEMENTS:
        count = text.count(old)
        print(f"  {label}: {count} match(es)")
        if count != 1:
            print(f"    EXPECTED exactly 1 match for {label}", file=sys.stderr)
            failed = True
            continue
        text = text.replace(old, new)

    if failed:
        print("\naborting; no change written", file=sys.stderr)
        return 1

    if CORRECTION_NOTE.strip() not in text:
        text = text.rstrip("\n") + "\n" + CORRECTION_NOTE
        print("  정정 주석: appended")

    print(f"\nresult: {len(row['markdown'])} -> {len(text)} chars")

    if not args.apply:
        print("dry run; pass --apply to write")
        return 0

    db_execute(
        "UPDATE research_documents SET markdown = %s, updated_at = NOW() WHERE id = %s",
        (text, row["id"]),
    )
    print("updated")
    clear_frontend_research_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
