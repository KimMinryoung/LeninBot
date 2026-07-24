"""Autonomous publication controls.

This module holds the policy that sits around public-bound autonomous output:
structural publication gates, optional pacing, and Stasova publication-security
review. Telegram channel broadcasts are intentionally not treated as an
external-platform tier here; the channel is an owned Cyber-Lenin distribution
surface.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from contextvars import ContextVar
from datetime import datetime
from urllib.parse import urlparse

from db import execute as db_execute, query_one as db_query_one
from telegram.channel_broadcast import current_autonomous_project_id

logger = logging.getLogger(__name__)

# Filenames of research drafts staged during the CURRENT autonomous tick.
# autonomous_project._run_one_tick initializes this to a fresh set per tick;
# outside the tick runtime it stays None so operator/task publication paths
# are unaffected. Used to enforce the cross-tick stage→publish gate: the
# context that wrote a draft must not be the context that fact-checks and
# publishes it.
current_tick_staged_slugs: ContextVar[set | None] = ContextVar(
    "current_tick_staged_slugs", default=None
)


def note_staged_this_tick(filename: str | None) -> None:
    staged = current_tick_staged_slugs.get()
    if staged is not None and filename:
        staged.add(str(filename))


def was_staged_this_tick(filename: str | None) -> bool:
    staged = current_tick_staged_slugs.get()
    return bool(staged) and str(filename or "") in staged

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        logger.warning("invalid integer env %s=%r; using %s", name, os.getenv(name), default)
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    logger.warning("invalid boolean env %s=%r; using %s", name, value, default)
    return default


PUBLICATION_PACING_ENABLED = _env_bool("AUTONOMOUS_PUBLICATION_PACING_ENABLED", False)
DEFAULT_MAX_PUBLICATIONS_PER_DAY = _env_int("AUTONOMOUS_MAX_PUBLICATIONS_PER_DAY", 3)
DEFAULT_COOLDOWN_AFTER_PUBLISH_MINUTES = _env_int(
    "AUTONOMOUS_COOLDOWN_AFTER_PUBLISH_MINUTES", 180
)

_AUDIT_TABLE_ENSURED = False


def _project_id() -> int | None:
    pid = current_autonomous_project_id.get()
    return int(pid) if pid is not None else None


def is_autonomous_publication_context() -> bool:
    """Return true when a publication is being attempted by an autonomous project."""
    return _project_id() is not None


_SOURCE_MARKER_RE = re.compile(
    r"https?://|KG:|knowledge_graph|vector_search|web_search|fetch_url",
    re.IGNORECASE,
)
_HTML_SECTION_RE = re.compile(r"<(?:article|section|h[1-3])(?:\s|>)", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_FENCED_CODE_RE = re.compile(r"```.*?```", re.DOTALL)


def _looks_like_http_url(value: str) -> bool:
    parsed = urlparse((value or "").strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _plain_text_from_html(html: str) -> str:
    return re.sub(r"\s+", " ", _TAG_RE.sub(" ", html or "")).strip()


def _source_marker_count(text: str) -> int:
    return len(_SOURCE_MARKER_RE.findall(text or ""))


def _format_gate_errors(kind: str, errors: list[str]) -> str:
    return (
        f"Autonomous publication blocked: {kind} does not meet the public quality gate.\n"
        + "\n".join(f"- {error}" for error in errors)
    )


def validate_autonomous_research_publication(
    *,
    title: str,
    content: str,
    identifier: str,
    fact_check_notes: str | None,
    is_edit: bool = False,
) -> str | None:
    """Hard quality gate for autonomous long-form public research.

    Structural format checks are mechanical only (heading levels, required
    section markers, footnote definitions) — semantic quality stays with the
    LLM review path. `is_edit` relaxes the fixed-layout checks so factual
    corrections to pre-layout legacy documents are not blocked.
    """
    text = (content or "").strip()
    notes = (fact_check_notes or "").strip()
    errors: list[str] = []
    source_count = _source_marker_count(notes)

    if not (title or "").strip():
        errors.append("title is required (got empty)")
    if not (identifier or "").strip():
        errors.append("stable slug/filename is required (got empty)")
    if source_count < 2:
        errors.append(
            "fact_check_notes must cite at least two source markers or URLs "
            f"(got {source_count}, need >= 2)"
        )

    prose = _FENCED_CODE_RE.sub("", text)
    body_h1s = re.findall(r"(?m)^\s*#\s+(\S.*)$", prose)
    if body_h1s:
        errors.append(
            f"body contains {len(body_h1s)} H1 heading(s) (e.g. '# {body_h1s[0][:60]}'); "
            "the document title is the only H1 — demote body section headings to ## / ###"
        )
    if not is_edit:
        if not re.search(r"(?m)^\s*##\s*요약\b", prose):
            errors.append(
                "body must contain a '## 요약' section (fixed report frame: 요약 first, "
                "출처 footnote definitions last; analysis sections between are free-form)"
            )
        footnote_defs = re.findall(r"(?m)^\s*\[\^[A-Za-z0-9_]+\]:", prose)
        if len(footnote_defs) < 2:
            errors.append(
                "body must define at least two footnote sources "
                f"('[^n]: publisher, title, date. URL' lines; got {len(footnote_defs)}, need >= 2)"
            )

    if errors:
        return _format_gate_errors("research", errors)
    return None


def validate_autonomous_hub_curation(
    *,
    title: str,
    source_url: str,
    source_title: str | None,
    source_publication: str | None,
    selection_rationale: str,
    context: str,
    slug: str,
    tags: list | None = None,
) -> str | None:
    """Hard quality gate for autonomous hub curation entries."""
    errors: list[str] = []

    if not (title or "").strip():
        errors.append("curation title is required (got empty)")
    if not _looks_like_http_url(source_url):
        errors.append(f"source_url must be a valid http(s) URL (got {source_url!r})")
    if not (source_title or "").strip():
        errors.append("source_title is required for autonomous curation (got empty)")
    if not (source_publication or "").strip():
        errors.append("source_publication is required for autonomous curation (got empty)")
    if not (selection_rationale or "").strip():
        errors.append("selection_rationale is required (got empty)")
    if not (context or "").strip():
        errors.append("context is required (got empty)")
    if not (slug or "").strip():
        errors.append("stable slug is required (got empty)")

    if errors:
        return _format_gate_errors("hub_curation", errors)
    return None


def validate_autonomous_static_page(
    *,
    slug: str,
    title: str,
    html_body: str,
    summary: str | None,
) -> str | None:
    """Hard quality gate for autonomous static pages."""
    plain = _plain_text_from_html(html_body or "")
    errors: list[str] = []

    if not (slug or "").strip():
        errors.append("stable slug is required (got empty)")
    if not (title or "").strip():
        errors.append("title is required (got empty)")
    if not (html_body or "").strip():
        errors.append("html_body is required (got empty)")
    if not plain:
        errors.append("page body must contain reader-visible text (got none)")
    if not _HTML_SECTION_RE.search(html_body or ""):
        errors.append("html_body must use semantic structure such as <article>, <section>, or <h2> (got none)")

    if errors:
        return _format_gate_errors("static_page", errors)
    return None


def ensure_publication_audit_table() -> None:
    global _AUDIT_TABLE_ENSURED
    if _AUDIT_TABLE_ENSURED:
        return
    db_execute("""
        CREATE TABLE IF NOT EXISTS autonomous_publication_audits (
            id SERIAL PRIMARY KEY,
            project_id INTEGER,
            publication_kind TEXT NOT NULL,
            title TEXT NOT NULL,
            public_url TEXT,
            content TEXT NOT NULL,
            review_report TEXT NOT NULL,
            warning_detected BOOLEAN NOT NULL DEFAULT TRUE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    db_execute("""
        CREATE INDEX IF NOT EXISTS autonomous_publication_audits_project_created_idx
        ON autonomous_publication_audits(project_id, created_at DESC)
    """)
    _AUDIT_TABLE_ENSURED = True


def _log_autonomous_event(project_id: int, event_type: str, content: str, meta: dict | None = None) -> None:
    try:
        db_execute(
            "INSERT INTO autonomous_project_events(project_id, event_type, content, meta) "
            "VALUES (%s, %s, %s, %s)",
            (project_id, event_type, content[:4000], json.dumps(meta or {})),
        )
    except Exception as exc:
        logger.warning("autonomous publication event log failed: %s", exc)


def check_autonomous_publication_allowed(publication_kind: str) -> tuple[bool, str]:
    """Return whether the current autonomous project may publish now.

    Non-autonomous callers are allowed; this control is scoped to scheduled
    autonomous projects via current_autonomous_project_id.
    """
    project_id = _project_id()
    if project_id is None:
        return True, "not an autonomous project publication"
    if not PUBLICATION_PACING_ENABLED:
        return True, "autonomous publication pacing disabled"

    try:
        row = db_query_one(
            "SELECT max_publications_per_day, cooldown_after_publish_minutes "
            "FROM autonomous_projects WHERE id = %s",
            (project_id,),
        ) or {}
    except Exception as exc:
        logger.warning("publication pacing config lookup failed: %s", exc)
        row = {}

    max_per_day = int(row.get("max_publications_per_day") or DEFAULT_MAX_PUBLICATIONS_PER_DAY)
    cooldown_minutes = int(
        row.get("cooldown_after_publish_minutes") or DEFAULT_COOLDOWN_AFTER_PUBLISH_MINUTES
    )

    count_row = db_query_one(
        """
        SELECT COUNT(*)::int AS n
          FROM autonomous_project_events
         WHERE project_id = %s
           AND event_type = 'publication_created'
           AND created_at >= NOW() - INTERVAL '24 hours'
        """,
        (project_id,),
    ) or {"n": 0}
    if max_per_day > 0 and int(count_row["n"]) >= max_per_day:
        return (
            False,
            f"Autonomous publication blocked: project #{project_id} already published "
            f"{count_row['n']}/{max_per_day} items in the last 24 hours.",
        )

    if cooldown_minutes > 0:
        last_row = db_query_one(
            """
            SELECT created_at
              FROM autonomous_project_events
             WHERE project_id = %s
               AND event_type = 'publication_created'
             ORDER BY created_at DESC
             LIMIT 1
            """,
            (project_id,),
        )
        if last_row and last_row.get("created_at"):
            elapsed = datetime.now(last_row["created_at"].tzinfo) - last_row["created_at"]
            remaining = cooldown_minutes - int(elapsed.total_seconds() // 60)
            if remaining > 0:
                return (
                    False,
                    f"Autonomous publication blocked: cooldown_after_publish has "
                    f"{remaining} minutes remaining for project #{project_id}.",
                )

    return True, "publication pacing check passed"


STASOVA_SPELLING_VERDICT_RE = re.compile(
    r"STASOVA_SPELLING_VERDICT:\s*(\{[^{}]*\})"
)


async def run_stasova_publication_review(
    *,
    publication_kind: str,
    title: str,
    content: str,
    public_url: str | None = None,
    spelling_corrections: list[str] | None = None,
) -> str:
    from agents import get_agent
    from bot_config import _get_task_provider
    from runtime_tools.registry import TOOL_HANDLERS as BASE_HANDLERS
    from runtime_tools.registry import TOOLS as BASE_TOOLS
    from telegram.bot import _get_model_for_agent, _make_provider_chat_fn

    spec = get_agent("stasova")
    agent_tools, agent_handlers = spec.filter_tools(BASE_TOOLS, BASE_HANDLERS)
    provider = spec.effective_provider(_get_task_provider())
    chat_fn = _make_provider_chat_fn(provider)
    project_id = _project_id()
    report_path = (
        f"temp_dev/stasova_reviews/autonomous_project_{project_id or 'manual'}_"
        f"{publication_kind}.md"
    )
    spelling_section = ""
    if spelling_corrections:
        spelling_section = (
            "\n\n부가 임무 — 표기 교정 검수:\n"
            "아래는 저장 시 사전 표준 표기로 자동 교정된 목록이다(직접 인용문은 이미 면제됨). "
            "각 교정이 문맥상 올바른지만 판단하라. 교정어가 문맥의 지시 대상과 다른 인물/개념을 "
            "가리키거나 원문 표기가 다른 단어의 일부였다면 그 교정은 취소 대상이다. "
            "이것은 기계적 표기 검수이지 문학적 편집이 아니다. 확신이 없으면 유지한다.\n"
            + "\n".join(spelling_corrections)
            + '\n\n점검 보고서의 마지막 줄에 반드시 다음 한 줄을 포함하라 (전부 올바르면 빈 배열):\n'
            'STASOVA_SPELLING_VERDICT: {"revert": [취소할 번호, ...]}'
        )
    review_task = (
        "다음 공개 발행물을 출판 보안 관점에서 점검하라.\n"
        "텔레그램 채널은 사이버-레닌이 공동 관리하는 공개 배포 채널이므로, "
        "그 사실 자체를 외부 플랫폼 리스크로 취급하지 말라.\n"
        "정치 노선 개정 판단이나 문학적 편집은 하지 말고, 시스템 프롬프트의 "
        "위험 축과 정치노선 왜곡 위험만 적용하라.\n"
        f"점검 보고서를 `{report_path}`에도 저장하라."
        f"{spelling_section}\n\n"
        f"발행 종류: {publication_kind}\n"
        f"공개 URL: {public_url or '(not known yet)'}\n\n"
        f"제목:\n{title}\n\n"
        f"본문:\n{content}"
    )
    return await chat_fn(
        [{"role": "user", "content": review_task}],
        system_prompt=spec.render_prompt(provider=provider),
        model=await _get_model_for_agent(spec),
        max_rounds=spec.max_rounds,
        max_tokens=4096,
        budget_usd=spec.budget_usd,
        extra_tools=agent_tools,
        extra_handlers=agent_handlers,
        task_id=None,
        agent_name="stasova",
        runtime_kind="autonomous",
    )


async def review_autonomous_publication(
    *,
    publication_kind: str,
    title: str,
    content: str,
    public_url: str | None = None,
    spelling_corrections: list[str] | None = None,
) -> str | tuple[str, list[int] | None]:
    """Run Stasova review for autonomous public-bound content and audit it.

    The review is advisory, matching Stasova's charter. It does not veto
    publication by itself; pacing controls are the hard gate.

    When `spelling_corrections` is passed (research publish path), the review
    additionally rules on each auto-applied spelling correction and the return
    becomes (note, revert_indices) — revert_indices is None when the review
    was skipped or produced no parsable verdict, so the caller can fall back
    to its own check.
    """
    project_id = _project_id()
    if project_id is None:
        note = "Stasova review: skipped (not an autonomous project publication)"
        return (note, None) if spelling_corrections is not None else note

    try:
        review_report = await run_stasova_publication_review(
            publication_kind=publication_kind,
            title=title,
            content=content,
            public_url=public_url,
            spelling_corrections=spelling_corrections,
        )
        from telegram.diary_publication import stasova_report_has_warning

        warning_detected = stasova_report_has_warning(review_report)
        ensure_publication_audit_table()
        row = await asyncio.to_thread(
            db_query_one,
            """
            INSERT INTO autonomous_publication_audits
                (project_id, publication_kind, title, public_url, content, review_report, warning_detected)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                project_id,
                publication_kind,
                title,
                public_url,
                content[:120000],
                review_report,
                warning_detected,
            ),
        )
        audit_id = int(row["id"]) if row else None
        _log_autonomous_event(
            project_id,
            "publication_reviewed",
            f"Stasova reviewed {publication_kind}: {title}",
            {"audit_id": audit_id, "warning_detected": warning_detected, "public_url": public_url},
        )
        warning_note = "warning detected" if warning_detected else "no warning detected"
        note = f"Stasova review: audit #{audit_id} ({warning_note})"
        if spelling_corrections is not None:
            reverts: list[int] | None = None
            verdict = STASOVA_SPELLING_VERDICT_RE.search(review_report or "")
            if verdict:
                try:
                    reverts = [
                        i for i in (json.loads(verdict.group(1)).get("revert") or [])
                        if isinstance(i, int)
                    ]
                except (ValueError, AttributeError):
                    reverts = None
            return note, reverts
        return note
    except Exception as exc:
        logger.error("Stasova autonomous publication review failed: %s", exc, exc_info=True)
        _log_autonomous_event(
            project_id,
            "publication_review_error",
            f"Stasova review failed for {publication_kind}: {exc}",
            {"public_url": public_url, "title": title},
        )
        note = f"Stasova review failed: {exc}"
        return (note, None) if spelling_corrections is not None else note


def record_autonomous_publication(
    *,
    publication_kind: str,
    title: str,
    public_url: str,
    meta: dict | None = None,
) -> None:
    project_id = _project_id()
    if project_id is None:
        return
    payload = {"kind": publication_kind, "title": title, "public_url": public_url}
    if meta:
        payload.update(meta)
    _log_autonomous_event(
        project_id,
        "publication_created",
        f"{publication_kind}: {title}\n{public_url}",
        payload,
    )


def record_autonomous_staged_draft(
    *,
    publication_kind: str,
    title: str,
    public_url: str,
    meta: dict | None = None,
) -> None:
    project_id = _project_id()
    if project_id is None:
        return
    note_staged_this_tick((meta or {}).get("filename"))
    payload = {"kind": publication_kind, "title": title, "public_url": public_url}
    if meta:
        payload.update(meta)
    _log_autonomous_event(
        project_id,
        "research_draft_staged",
        f"{publication_kind}: {title}\n{public_url}",
        payload,
    )
