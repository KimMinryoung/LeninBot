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


def test_research_gate() -> None:
    bad = validate_autonomous_research_publication(
        title="짧음",
        identifier="x.md",
        content="짧은 본문",
        fact_check_notes="확인함",
    )
    _assert_blocked(bad, "source markers")

    good = validate_autonomous_research_publication(
        title="짧은 보고서",
        identifier="20260509-industrial-policy.md",
        content="짧아도 구조적으로는 발행 도구가 처리할 수 있는 본문",
        fact_check_notes="https://example.com/a 와 https://example.com/b 확인",
    )
    assert good is None, good

    procedural_use = validate_autonomous_research_publication(
        title="협동조합 설립 절차",
        identifier="cooperative-practice-guide.md",
        content=(
            "행정 담당자는 정관 초안 취합, 관공서 대응, 조합 명의 임시 계좌 개설을 맡는다."
        ),
        fact_check_notes="https://example.com/law 와 https://example.com/tax 확인",
    )
    assert procedural_use is None, procedural_use


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
    test_hub_gate()
    test_static_page_gate()
    print("autonomous publication gates smoke ok")


if __name__ == "__main__":
    main()
