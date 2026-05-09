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
        content="TODO: 초안",
        fact_check_notes="https://example.com 확인",
    )
    _assert_blocked(bad, "public quality gate")

    section = "## 분석\n" + ("2026년 현재 기준의 효용을 따져 낡은 부분을 수정했다. " * 25)
    content = section + "\n\n## 근거\n" + ("구체적 논지와 현실 적용 가능성을 검토했다. " * 35)
    good = validate_autonomous_research_publication(
        title="2026년 현재 조건에서의 산업정책 보고서",
        identifier="20260509-industrial-policy.md",
        content=content,
        fact_check_notes=(
            "https://example.com/a 와 https://example.com/b 를 대조했다. "
            "2026년 현재 기준으로 낡은 통계와 낮은 효용의 전망 문장을 수정했고, "
            "현재 유지할 수 있는 주장과 삭제한 주장을 구분했다. "
            "KG:autonomous_project_test 에서 기존 내부 메모도 함께 확인했다. "
            "출처 간 불일치가 있는 수치는 본문에서 제거했고, 실천적 판단에 도움이 되는 "
            "최근 제도 조건과 독자가 재검증할 수 있는 근거만 남겼다."
        ),
    )
    assert good is None, good


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
        title="2026년 현재에도 읽을 가치가 있는 현장 분석",
        source_url="https://example.com/article",
        source_title="현장 조직화의 실제 쟁점",
        source_publication="예시매체",
        selection_rationale=(
            "2026년 현재 기준으로도 현장 조직화의 병목을 구체적으로 보여주며, "
            "이론적 주장과 실제 사례가 함께 있어 큐레이션 가치가 충분하다. "
            "낡은 전망을 반복하지 않고 독자가 현장 판단에 바로 쓸 수 있는 기준을 제시한다."
        ),
        context=(
            "최근 논쟁에서 추상적 구호가 반복되는 문제를 보완한다. 낡은 전망은 배제하고 "
            "현재 독자가 바로 비교할 수 있는 쟁점을 중심에 두며, 후속 연구의 출발점으로도 유용하다."
        ),
        slug="current-field-analysis",
        tags=["현장", "조직"],
    )
    assert good is None, good


def test_static_page_gate() -> None:
    bad = validate_autonomous_static_page(
        slug="x",
        title="임시",
        html_body="<p>TODO</p>",
        summary=None,
    )
    _assert_blocked(bad, "stable slug")

    paragraph = "2026년 현재 기준에서 독자가 바로 활용할 수 있도록 낡은 설명을 고치고 핵심 쟁점을 정리했다. "
    html = "<article><section><h2>현재 효용</h2>" + "".join(
        f"<p>{paragraph}</p>" for _ in range(18)
    ) + "</section></article>"
    good = validate_autonomous_static_page(
        slug="current-reference-page",
        title="2026년 현재 기준 참고 페이지",
        html_body=html,
        summary=(
            "2026년 현재 기준으로 낮은 효용의 설명을 제거하고 독자가 바로 활용할 수 있는 "
            "쟁점과 검토 기준을 정리한 참고 페이지이며, 낡은 전망을 현재 조건에 맞게 고쳤다."
        ),
    )
    assert good is None, good


def main() -> None:
    test_research_gate()
    test_hub_gate()
    test_static_page_gate()
    print("autonomous publication gates smoke ok")


if __name__ == "__main__":
    main()
