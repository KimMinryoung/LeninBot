"""Callback display cropping must not hide an ordinary briefing's findings."""
from telegram.task_reporting import REPORT_CONTEXT_CHARS, report_for_callback


def test_briefing_after_long_collection_preamble_reaches_orchestrator():
    report = "수집 상태 설명 " * 600 + "Jacobin: 제국 논쟁. Rundown: AI 안전성 논쟁."
    assert 3000 < len(report) < REPORT_CONTEXT_CHARS
    assert report_for_callback(1348, report) == report


def test_oversized_report_has_exact_saved_result_continuation():
    report = "가" * REPORT_CONTEXT_CHARS + "누락된 내용"
    rendered = report_for_callback(1348, report)
    assert rendered.startswith(report[:REPORT_CONTEXT_CHARS])
    assert "누락된 내용" not in rendered
    assert "id=1348, offset=24000, max_chars=12000" in rendered
    assert "this says nothing about task completion" in rendered
    assert "in this turn" in rendered


def test_empty_and_exact_limit_reports_are_not_cropped():
    for report in ("", "x" * REPORT_CONTEXT_CHARS):
        assert report_for_callback(7, report) == report
