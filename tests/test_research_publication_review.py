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


@pytest.mark.parametrize("response", ["", "PASS", '{"verdict":"PASS"}',
    '{"verdict":"PASS","reason":"ok","issues":["unsupported"]}', "```json\n{}\n```"])
def test_ambiguous_review_never_passes(response):
    assert review.parse_review(response)["verdict"] == "UNVERIFIED"


def test_review_exception_saved_as_unverified(monkeypatch, tmp_path):
    monkeypatch.setattr(review, "REVIEW_DIR", tmp_path)
    monkeypatch.setattr(review, "_run_review", AsyncMock(side_effect=RuntimeError("provider unavailable")))
    result = asyncio.run(review.review_research_document(document=BODY))
    assert result["verdict"] == "UNVERIFIED"
    stored = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert stored["document_sha256"] == hashlib.sha256(BODY.encode()).hexdigest()


def test_review_does_not_replace_author_context(monkeypatch, tmp_path):
    from provenance.runtime import init_provenance_buffer, get_provenance_buffer
    monkeypatch.setattr(review, "REVIEW_DIR", tmp_path)

    async def fake(*args):
        init_provenance_buffer(agent="task_verifier")
        return {"verdict": "PASS", "reason": "Source checked", "issues": []}, {}

    monkeypatch.setattr(review, "_run_review", fake)

    async def run():
        author = init_provenance_buffer(agent="analyst")
        await review.review_research_document(document=BODY)
        assert get_provenance_buffer() is author

    asyncio.run(run())


@pytest.mark.parametrize("read_source,interrupted", [(False, False), (True, False), (True, True)])
def test_reviewer_surface_and_evidence_receipt(monkeypatch, read_source, interrupted):
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

    async def chat(messages, **kw):
        assert {t["name"] for t in kw["extra_tools"]} == {"fetch_url"}
        assert set(kw["extra_handlers"]) == {"fetch_url"}
        assert json.loads(messages[0]["content"])["candidate"] == BODY
        if read_source:
            await kw["extra_handlers"]["fetch_url"](url="https://example.org/source")
        kw["budget_tracker"]["final_response"] = json.dumps({"verdict": "PASS", "reason": "Source checked", "issues": []})
        kw["budget_tracker"]["final_response_truncated"] = interrupted
        return "progress is not the verdict"

    monkeypatch.setitem(sys.modules, "telegram.bot", SimpleNamespace(_make_provider_chat_fn=lambda _: chat))
    evidence = []
    verdict, _ = asyncio.run(review._run_review(BODY, NOTES, evidence))
    assert verdict["verdict"] == ("PASS" if read_source and not interrupted else "UNVERIFIED")
    if read_source:
        assert evidence[0]["sha256"] == hashlib.sha256(b"Observed source text").hexdigest()


def test_autonomous_keeps_existing_review_contract(isolated, monkeypatch):
    monkeypatch.setattr(r, "is_autonomous_publication_context", lambda: True)
    mocked = AsyncMock(side_effect=AssertionError("duplicate task review"))
    monkeypatch.setattr(r, "review_research_document", mocked)
    assert asyncio.run(r._review_before_public_write(BODY, NOTES)) == ""
    mocked.assert_not_awaited()
