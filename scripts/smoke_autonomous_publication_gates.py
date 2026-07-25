#!/usr/bin/env python3
"""Smoke checks for autonomous publication quality gates."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autonomous_publication_controls import (
    validate_autonomous_hub_curation,
    validate_autonomous_research_publication,
    validate_autonomous_static_page,
)


def _assert_blocked(message: str | None, needle: str) -> None:
    assert message is not None, "expected gate to block"
    assert needle in message, message


# The gate requires the fixed report frame, so any body meant to pass needs it.
# These fixtures predated the 요약 and footnote rules and were asserting a pass on
# a body that has neither, which left this smoke check failing on HEAD.
FRAME = """## 요약
핵심 결론을 먼저 쓴다. [확정] 지표는 이렇다.

## 분석
본문 한 문단.

[^1]: 연합뉴스, 기사 제목, 2026-07-01. https://example.com/a
[^2]: 머니투데이, 기사 제목, 2026-07-02. https://example.com/b
"""

NOTES = "https://example.com/a 와 https://example.com/b 확인"


def _research(content: str, **kwargs):
    return validate_autonomous_research_publication(
        title=kwargs.pop("title", "보고서 제목"),
        identifier=kwargs.pop("identifier", "20260509-industrial-policy.md"),
        content=content,
        fact_check_notes=kwargs.pop("fact_check_notes", NOTES),
        **kwargs,
    )


def test_research_gate() -> None:
    bad = validate_autonomous_research_publication(
        title="짧음",
        identifier="x.md",
        content="짧은 본문",
        fact_check_notes="확인함",
    )
    _assert_blocked(bad, "source markers")

    assert _research(FRAME) is None, _research(FRAME)

    procedural_use = _research(
        "행정 담당자는 정관 초안 취합, 관공서 대응, 조합 명의 임시 계좌 개설을 맡는다.\n\n" + FRAME,
        title="협동조합 설립 절차",
        identifier="cooperative-practice-guide.md",
    )
    assert procedural_use is None, procedural_use

    _assert_blocked(_research("# 본문 안의 H1\n\n" + FRAME), "H1 heading")
    _assert_blocked(_research("본문만 있고 프레임이 없다"), "요약")


def test_research_related_report_line() -> None:
    """The related-report line is the only same-tab internal link the frame has.

    A bare path renders its markdown source to the reader, an absolute self link
    opens a new tab, and a slug that does not exist publishes a 404, so all three
    are mechanical failures rather than prompt guidance.
    """
    canonical = "**선행 보고서:** [제목](/reports/research/some-slug)\n\n" + FRAME
    # The slug existence check needs the database. Without one the gate skips it,
    # so accept either outcome and assert only the shape here.
    shape_only = _research(canonical)
    assert shape_only is None or "do not exist" in shape_only, shape_only

    _assert_blocked(
        _research("**선행 보고서:** /reports/research/some-slug\n\n" + FRAME),
        "related-report line must read exactly",
    )
    _assert_blocked(
        _research("**선행보고서** [제목](/reports/research/some-slug)\n\n" + FRAME),
        "related-report line must read exactly",
    )
    _assert_blocked(
        _research("**선행 보고서:** [가](/reports/research/a), [나](/reports/research/b)\n\n" + FRAME),
        "related-report line must read exactly",
    )
    _assert_blocked(
        _research("본문에서 [옛 보고서](https://cyber-lenin.com/reports/research/x)를 인용한다.\n\n" + FRAME),
        "root-relative",
    )


def test_hub_gate() -> None:
    bad = validate_autonomous_hub_curation(
        title="좋은 글",
        source_url="https://example.com/article",
        source_title=None,
        source_publication=None,
        selection_rationale="좋다",
        context="짧다",
        slug="hub-test",
    )
    _assert_blocked(bad, "source_title")

    good = validate_autonomous_hub_curation(
        title="짧은 큐레이션",
        source_url="https://example.com/article",
        source_title="짧은 글",
        source_publication="예시매체",
        selection_rationale="좋다",
        context="짧다",
        slug="current-field-analysis",
        tags=["현장", "조직"],
    )
    assert good is None, good


def test_static_page_gate() -> None:
    bad = validate_autonomous_static_page(
        slug="x",
        title="짧음",
        html_body="",
        summary=None,
    )
    _assert_blocked(bad, "html_body is required")

    good = validate_autonomous_static_page(
        slug="current-reference-page",
        title="실천 참고 페이지",
        html_body="<article><section><h2>검토 기준</h2><p>짧은 본문</p></section></article>",
        summary=None,
    )
    assert good is None, good


def main() -> None:
    test_research_gate()
    test_research_related_report_line()
    test_hub_gate()
    test_static_page_gate()
    print("autonomous publication gates smoke ok")


if __name__ == "__main__":
    main()
