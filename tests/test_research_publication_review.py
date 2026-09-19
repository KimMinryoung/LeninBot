"""Publication/revision regression cases from analyst task #1347 (no live I/O)."""
import asyncio
import hashlib
import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from runtime_tools import research as r
from runtime_tools import research_review as review
from tool_gateway.results import ToolFailure


BODY = "## 분석\n원문의 자기 진술이다.[^1]\n\n[^1]: 원문 https://example.org/source"
NOTES = "원문과 인용 및 자기 진술의 범위를 대조했다. " * 8 + "https://example.org/source"


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    writes = []
    current = {"id": 457, "slug": "report", "filename": "report.md", "title": "보고서",
               "status": "staged", "markdown": r._build_document("보고서", BODY, "2026-09-15")}
    monkeypatch.setattr(r, "PUBLICATION_DRAFT_DIR", tmp_path / "drafts")
    monkeypatch.setattr(review, "REVIEW_DIR", tmp_path / "reviews")
    monkeypatch.setattr(r, "is_autonomous_publication_context", lambda: False)
    monkeypatch.setattr(r, "_mechanical_spelling_pass", lambda _: None)
    monkeypatch.setattr(r, "_normalize_standard_spellings", AsyncMock(side_effect=lambda text: (text, "")))
    monkeypatch.setattr(r, "check_autonomous_publication_allowed", lambda _: (True, ""))
    monkeypatch.setattr(r, "review_autonomous_publication", AsyncMock(return_value=""))
    monkeypatch.setattr(r, "record_autonomous_publication", Mock())
    monkeypatch.setattr(r, "record_autonomous_staged_draft", Mock())
    monkeypatch.setattr(r.research_store, "get_document", lambda *a, **kw: dict(current))

    def write(**kw):
        writes.append(kw)
        current.update(kw, content_sha256=hashlib.sha256(kw["markdown"].encode()).hexdigest())
        return dict(current), False

    monkeypatch.setattr(r.research_store, "upsert_document", write)
    monkeypatch.setattr(r.research_store, "set_status", Mock(side_effect=AssertionError("unreviewed visibility write")))
    monkeypatch.setattr(r, "_invalidate_cache_sync", Mock(return_value={"ok": True, "deleted": 0}))
    monkeypatch.setattr(r, "_purge_cloudflare_sync", Mock(return_value={"ok": True}))
    monkeypatch.setattr(r, "_format_invalidation_note", lambda *a, **kw: "cache checked")
    monkeypatch.setattr(r, "maybe_broadcast_autonomous_publication", AsyncMock())
    return current, writes


def receipt(document, verdict="PASS"):
    return {"verdict": verdict, "reason": "Checked source", "issues": [] if verdict == "PASS" else ["Unsupported attribution"],
            "document_sha256": hashlib.sha256(document.encode()).hexdigest(), "receipt_path": "/tmp/review.json"}


@pytest.mark.parametrize("verdict", ["REVISE", "UNVERIFIED"])
def test_blocked_review_precedes_all_public_side_effects(isolated, monkeypatch, verdict):
    current, writes = isolated
    monkeypatch.setattr(r, "review_research_document", AsyncMock(side_effect=lambda **kw: receipt(kw["document"], verdict)))
    result = asyncio.run(r._exec_research_document(action="publish_public", slug="report", fact_check_notes=NOTES))
    assert isinstance(result, ToolFailure)
    assert not writes and current["status"] == "staged"
    r._invalidate_cache_sync.assert_not_called()
    r._purge_cloudflare_sync.assert_not_called()
    r.maybe_broadcast_autonomous_publication.assert_not_awaited()


def test_pass_publishes_exact_reviewed_document(isolated, monkeypatch):
    current, writes = isolated
    seen = []

    async def inspect(**kw):
        assert not writes
        seen.append(kw["document"])
        return receipt(kw["document"])

    monkeypatch.setattr(r, "review_research_document", inspect)
    result = asyncio.run(r._exec_research_document(action="publish_public", slug="report", fact_check_notes=NOTES, broadcast=False))
    assert not isinstance(result, ToolFailure)
    assert writes[0]["status"] == "public"
    assert writes[0]["markdown"] == seen[0]
    assert "Independent document review: PASS" in result


def test_wrong_hash_cannot_authorize_write(isolated, monkeypatch):
    monkeypatch.setattr(r, "review_research_document", AsyncMock(return_value=receipt("different document")))
    result = asyncio.run(r._exec_research_document(action="publish_public", slug="report", fact_check_notes=NOTES))
    assert isinstance(result, ToolFailure)
    assert not isolated[1]


@pytest.mark.parametrize("action", ["edit_public", "republish_public", "publish_private"])
def test_alternative_public_paths_are_reviewed(isolated, monkeypatch, action):
    from runtime_tools import private_reports
    current, writes = isolated
    current["status"] = "public" if action == "edit_public" else "private"
    monkeypatch.setattr(private_reports, "get_private_report_sync", lambda **kw: dict(current))
    monkeypatch.setattr(r, "review_research_document", AsyncMock(side_effect=lambda **kw: receipt(kw["document"], "REVISE")))
    result = asyncio.run(r._exec_research_document(action=action, slug="report", content=BODY, fact_check_notes=NOTES))
    assert isinstance(result, ToolFailure)
    assert not writes
    r.maybe_broadcast_autonomous_publication.assert_not_awaited()


def test_failed_batch_is_atomic_and_returns_tool_failure(isolated):
    result = asyncio.run(r._exec_research_document(action="edit_staged", slug="report", edits=[
        {"find": "자기 진술", "replace": "본인 진술"},
        {"find": "**작성일:** 2026-09-15", "replace": ""},
    ]))
    assert isinstance(result, ToolFailure)
    assert "No edits were applied" in result and "generated title/author/date" in result
    assert not isolated[1]
    assert "자기 진술" in isolated[0]["markdown"]


def test_multiline_body_edit_succeeds(isolated):
    result = asyncio.run(r._exec_research_document(action="edit_staged", slug="report", edits=[
        {"find": "## 분석\n원문의 자기 진술이다.[^1]", "replace": "## 분석\n본인이 그렇게 진술했다.[^1]"},
    ]))
    assert not isinstance(result, ToolFailure)
    assert isolated[1][0]["status"] == "staged"
    assert "본인이 그렇게 진술했다" in isolated[0]["markdown"]


def test_staged_read_matches_editable_body(isolated, monkeypatch):
    import db
    from self_runtime.tools import _exec_read_research
    monkeypatch.setattr(db, "query_one", lambda *a, **kw: dict(isolated[0]))
    result = asyncio.run(_exec_read_research(slug="report", max_chars=1000))
    assert "view=editable_body" in result
    assert result.endswith(BODY)
    assert "**작성일:**" not in result


def test_header_normalization_handles_observed_duplicate():
    candidate = r._build_document("보고서", "**작성:** Cyber-Lenin | **작성일:** 2026-09-15\n\n---\n\n" + BODY, "2026-09-15")
    assert r._strip_leading_research_scaffold(candidate) == BODY
    assert r._strip_leading_research_scaffold(r._build_document("보고서", "---\n\n" + BODY, "2026-09-15")) == BODY


def test_review_exception_saved_as_unverified(monkeypatch, tmp_path):
    monkeypatch.setattr(review, "REVIEW_DIR", tmp_path)
    monkeypatch.setattr(review, "_run_review", AsyncMock(side_effect=RuntimeError("provider unavailable")))
    result = asyncio.run(review.review_research_document(document=BODY))
    assert result["verdict"] == "UNVERIFIED" and result["failure_kind"] == "review_unavailable"
    stored = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert stored["document_sha256"] == hashlib.sha256(BODY.encode()).hexdigest()
    assert stored["document"] == BODY  # blocked candidates are kept; PASS text lives in the DB


def test_review_does_not_replace_author_context(monkeypatch, tmp_path):
    from provenance.runtime import init_provenance_buffer, get_provenance_buffer
    monkeypatch.setattr(review, "REVIEW_DIR", tmp_path)

    async def fake(*args, **kwargs):
        init_provenance_buffer(agent="task_verifier")
        return {"verdict": "PASS", "reason": "Source checked", "issues": []}, {}, ""

    monkeypatch.setattr(review, "_run_review", fake)

    async def run():
        author = init_provenance_buffer(agent="analyst")
        await review.review_research_document(document=BODY)
        assert get_provenance_buffer() is author

    asyncio.run(run())


def install_reviewer(monkeypatch, chat):
    import bot_config
    import llm.runtime_profile
    from runtime_tools import registry

    monkeypatch.setattr(bot_config, "_get_task_provider", lambda: "deepseek")
    monkeypatch.setattr(llm.runtime_profile, "resolve_runtime_profile", AsyncMock(return_value=SimpleNamespace(model_id="fake-low")))
    monkeypatch.setattr(registry, "TOOLS", [
        {"name": "fetch_url"}, {"name": "research_document"}, {"name": "send_email"},
    ])
    monkeypatch.setattr(registry, "TOOL_HANDLERS", {
        "fetch_url": AsyncMock(return_value="Observed source text"),
        "research_document": AsyncMock(), "send_email": AsyncMock(),
    })
    monkeypatch.setitem(sys.modules, "telegram.bot", SimpleNamespace(_make_provider_chat_fn=lambda _: chat))


VERDICT = review.VERDICT_TOOL["name"]


def test_reviewer_surface_records_verdict_and_evidence(monkeypatch):
    kw_seen = {}

    async def chat(messages, **kw):
        kw_seen.update(kw)
        assert {t["name"] for t in kw["extra_tools"]} == {"fetch_url", VERDICT}
        assert set(kw["extra_handlers"]) == {"fetch_url", VERDICT}
        assert kw["terminal_tools"] == kw["finalization_tools"] == [VERDICT] and kw["terminal_required"]
        assert kw["budget_usd"] == review.REVIEW_BUDGET_USD and kw["max_rounds"] == review.REVIEW_MAX_ROUNDS
        assert json.loads(messages[0]["content"])["candidate"] == BODY
        await kw["extra_handlers"]["fetch_url"](url="https://example.org/source")
        assert await kw["extra_handlers"][VERDICT](verdict="PASS", reason="Source checked", issues=[]) == "Verdict recorded: PASS"
        kw["budget_tracker"].update(total_cost=0.01, rounds_used=2, final_response="prose is ignored")
        return "prose is ignored"

    install_reviewer(monkeypatch, chat)
    evidence = []
    verdict, usage, final_text = asyncio.run(review._run_review(BODY, NOTES, evidence))
    assert verdict == {"verdict": "PASS", "reason": "Source checked", "issues": []}
    assert usage == {"provider": "deepseek", "model": "fake-low", "total_cost": 0.01, "rounds_used": 2}
    assert final_text == ""
    assert "older version" in kw_seen["system_prompt"]
    assert evidence[0]["sha256"] == hashlib.sha256(b"Observed source text").hexdigest()


@pytest.mark.parametrize("read_source,args,message", [
    (True, {"verdict": "PASS", "reason": "ok", "issues": ["unsupported"]}, "PASS cannot list"),
    (True, {"verdict": "REVISE", "reason": "bad", "issues": []}, "REVISE requires"),
    (True, {"verdict": "REVISE", "reason": "bad", "issues": ["  "]}, "REVISE requires"),
    (True, {"verdict": "PASS", "reason": " ", "issues": []}, "reason must"),
    (False, {"verdict": "PASS", "reason": "ok", "issues": []}, "PASS requires at least one"),
])
def test_verdict_tool_rejects_inconsistent_verdicts(read_source, args, message):
    from tool_gateway.results import ToolRejection
    evidence = [{"tool": "fetch_url", "error": not read_source}]
    box = {}
    with pytest.raises(ToolRejection, match=message):
        asyncio.run(review.make_verdict_recorder(evidence, box)(**args))
    assert not box


def test_verdict_tool_records_once():
    from tool_gateway.results import ToolRejection
    box = {}
    record = review.make_verdict_recorder([], box)
    asyncio.run(record(verdict="UNVERIFIED", reason="source 401", issues=[]))
    with pytest.raises(ToolRejection, match="already recorded"):
        asyncio.run(record(verdict="REVISE", reason="changed my mind", issues=["x"]))
    assert box["verdict"] == "UNVERIFIED"


def test_prose_without_verdict_is_unverified_and_kept_for_diagnosis(monkeypatch, tmp_path):
    monkeypatch.setattr(review, "REVIEW_DIR", tmp_path)

    async def chat(messages, **kw):
        await kw["extra_handlers"]["fetch_url"](url="https://example.org/source")
        kw["budget_tracker"]["final_response"] = '{"verdict": "PASS", "reason": "in prose", "issues": []}'
        return "ignored"

    install_reviewer(monkeypatch, chat)
    result = asyncio.run(review.review_research_document(document=BODY))
    assert result["verdict"] == "UNVERIFIED" and result["failure_kind"] == "no_verdict"
    stored = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert stored["final_text"].startswith('{"verdict": "PASS"')
    assert stored["evidence"][0]["tool"] == "fetch_url"


def test_deadline_is_unverified(monkeypatch, tmp_path):
    monkeypatch.setattr(review, "REVIEW_DIR", tmp_path)
    monkeypatch.setattr(review, "REVIEW_DEADLINE_SECONDS", 0.05)

    async def chat(messages, **kw):
        await kw["extra_handlers"]["fetch_url"](url="https://example.org/source")
        await asyncio.sleep(5)

    install_reviewer(monkeypatch, chat)
    result = asyncio.run(review.review_research_document(document=BODY))
    assert result["failure_kind"] == "review_unavailable" and "TimeoutError" in result["reason"]
    assert json.loads(next(tmp_path.glob("*.json")).read_text())["evidence"]


def test_unsaved_receipt_withdraws_pass_but_keeps_findings(monkeypatch, tmp_path):
    monkeypatch.setattr(review, "REVIEW_DIR", tmp_path / "missing.json")
    (tmp_path / "missing.json").write_text("not a directory")

    async def chat(messages, **kw):
        await kw["extra_handlers"][VERDICT](verdict="REVISE", reason="bad claim", issues=["fix attribution"])

    install_reviewer(monkeypatch, chat)
    result = asyncio.run(review.review_research_document(document=BODY))
    assert result["verdict"] == "UNVERIFIED" and result["failure_kind"] == "receipt_unavailable"
    assert result["issues"] == ["fix attribution"] and "receipt_path" not in result


def test_autonomous_keeps_existing_review_contract(isolated, monkeypatch):
    monkeypatch.setattr(r, "is_autonomous_publication_context", lambda: True)
    mocked = AsyncMock(side_effect=AssertionError("duplicate task review"))
    monkeypatch.setattr(r, "review_research_document", mocked)
    assert asyncio.run(r._review_before_public_write(BODY, NOTES)) == ""
    mocked.assert_not_awaited()


@pytest.mark.parametrize("verdict,expected,forbidden", [
    ("REVISE", "edit_staged", "unchanged"),
    ("UNVERIFIED", "edit_staged", "unchanged"),
])
def test_content_verdicts_direct_the_author_to_the_findings(isolated, monkeypatch, verdict, expected, forbidden):
    monkeypatch.setattr(r, "review_research_document", AsyncMock(side_effect=lambda **kw: receipt(kw["document"], verdict)))
    message = asyncio.run(r._exec_research_document(action="publish_public", slug="report", fact_check_notes=NOTES))
    assert isinstance(message, ToolFailure)
    assert expected in message and forbidden not in message
    assert not isolated[1]


def test_incomplete_review_does_not_instruct_body_edits(isolated, monkeypatch):
    result = receipt(isolated[0]["markdown"], "UNVERIFIED")
    result.update(review._unverified("no_verdict", "Reviewer ended without recording a verdict."))
    monkeypatch.setattr(r, "review_research_document", AsyncMock(return_value=result))
    message = asyncio.run(r._exec_research_document(action="publish_public", slug="report", fact_check_notes=NOTES))
    assert isinstance(message, ToolFailure)
    assert "Retry the same publication call unchanged" in message and "edit_staged" not in message
    assert not isolated[1]


def test_verdict_tool_is_registered_with_the_gateway():
    from security_gateway.policy import risk_class
    assert risk_class(VERDICT) == "state"


def _write_receipt(tmp_path, name, **fields):
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps({"verdict": "REVISE", "reason": "old", "issues": ["fix x"],
                                "slug": "report.md", "reviewed_at": "2026-09-19T00:00:00+00:00", **fields}))
    return path


@pytest.mark.parametrize("prior,expected", [
    (None, None),
    ({}, ["fix x"]),
    ({"verdict": "PASS", "issues": []}, None),
    ({"slug": "other.md"}, None),
    ({"failure_kind": "no_verdict", "issues": []}, None),
])
def test_resubmission_hands_previous_blocked_findings_to_the_reviewer(monkeypatch, tmp_path, prior, expected):
    monkeypatch.setattr(review, "REVIEW_DIR", tmp_path)
    if prior is not None:
        _write_receipt(tmp_path, "prior", **prior)
    seen = {}

    async def chat(messages, **kw):
        seen.update(json.loads(messages[0]["content"]))
        await kw["extra_handlers"][VERDICT](verdict="UNVERIFIED", reason="not enough", issues=[])

    install_reviewer(monkeypatch, chat)
    result = asyncio.run(review.review_research_document(document=BODY, slug="report.md"))
    assert result["slug"] == "report.md"
    if expected is None:
        assert "previous_review" not in seen
    else:
        assert seen["previous_review"]["issues"] == expected and seen["previous_review"]["verdict"] == "REVISE"


def test_latest_receipt_for_slug_wins(monkeypatch, tmp_path):
    import os
    monkeypatch.setattr(review, "REVIEW_DIR", tmp_path)
    old = _write_receipt(tmp_path, "old", issues=["old finding"])
    os.utime(old, (1, 1))
    _write_receipt(tmp_path, "new", verdict="PASS", issues=[])
    assert review._previous_findings("report.md") is None
    (tmp_path / "new.json").unlink()
    assert review._previous_findings("report.md")["issues"] == ["old finding"]


def test_public_paths_pass_the_slug_to_the_review(isolated, monkeypatch):
    slugs = []

    async def inspect(**kw):
        slugs.append(kw.get("slug"))
        return receipt(kw["document"], "REVISE")

    monkeypatch.setattr(r, "review_research_document", inspect)
    asyncio.run(r._exec_research_document(action="publish_public", slug="report", fact_check_notes=NOTES))
    isolated[0]["status"] = "public"
    asyncio.run(r._exec_research_document(action="edit_public", slug="report", content=BODY, fact_check_notes=NOTES))
    assert slugs == ["report.md", "report.md"]
