"""`/curate <url> [note]` — owner-commissioned /hub curation from the Telegram bot.

The command validates and deduplicates the link, then enqueues a `telegram_tasks`
row with `agent_type="hub_curator"`. The regular task worker runs the agent
(`agents/hub_curator.py`); this module supplies the two hooks the worker needs
around that run:

- `make_guarded_publish_handler` wraps `publish_hub_curation` so the field rules
  are enforced deterministically at the write boundary, and so the tool's plain
  "Error: ..." strings become `ToolFailure` (otherwise the agent loop's
  terminal-tool short-circuit would treat a rejected publish as success).
- `report_curation_outcome` sends the owner a deterministic result DM based on
  whether a `hub_curations` row for the URL exists after the run, not on what the
  agent said.

This module must not import `telegram.bot` (the bot imports it).
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
from typing import Any, Awaitable, Callable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from content_fetch.url_security import UnsafeUrlError
from content_fetch.url_security import validate_public_http_url as _validate_public_http_url
from db import execute as _execute, query as _query, query_one as _query_one
from tool_gateway.results import ToolFailure

logger = logging.getLogger(__name__)

AGENT_NAME = "hub_curator"
HUB_PUBLIC_URL = "https://cyber-lenin.com/hub/{slug}"
USAGE_TEXT = (
    "사용법: /curate <url> [메모]\n"
    "링크가 든 메시지에 답장하면서 /curate 만 보내도 된다. "
    "메모는 이 글을 고른 이유나 강조할 각도를 에이전트에 전달한다."
)

# Same shape as content_fetch.urls._URL_PATTERN; a local copy keeps this module
# free of the browser-pool imports that content_fetch.urls drags in.
_URL_RE = re.compile(r'https?://[^\s<>"\')]+')
_COMMAND_RE = re.compile(r"^\s*/curate(?:@\w+)?\b", re.IGNORECASE)
_TRACKING_PARAM_RE = re.compile(
    r"^(utm_\w+|fbclid|gclid|dclid|gbraid|wbraid|igshid|mc_cid|mc_eid|ref_src|_ga)$",
    re.IGNORECASE,
)
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,79}$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_HANGUL_RE = re.compile(r"[가-힣]")
_MARKDOWN_LINE_START_RE = re.compile(r"^\s*(#|[-*]\s|\d+\.\s|>)")
_MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\([^)]+\)")
_META_TITLE_RE = re.compile(r"^\s*(큐레이션|추천|왜 이 글이|이 글은)|#\s?\d")
_ACTIVE_TASK_STATUSES = ("pending", "queued", "processing")


# ── URL handling ─────────────────────────────────────────────────────────


def normalize_source_url(url: str) -> str:
    """Canonical form used for duplicate detection.

    Lowercases scheme and host, drops `www.`, default ports, the fragment, and
    tracking query parameters, sorts the remaining query, and strips a trailing
    slash from a non-root path. Never used for fetching: the agent receives the
    URL exactly as the owner sent it.
    """
    parts = urlsplit((url or "").strip())
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    port = parts.port
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        host = f"{host}:{port}"
    path = parts.path.rstrip("/") or "/"
    query_pairs = sorted(
        (k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not _TRACKING_PARAM_RE.match(k)
    )
    query = urlencode(query_pairs, doseq=True)
    return urlunsplit((scheme, host, path, query, ""))


def parse_curate_args(
    text: str | None,
    reply_text: str | None = None,
    entity_urls: list[str] | None = None,
) -> tuple[str | None, str]:
    """Return (url, note) from the command text, falling back to link entities
    and then to the replied-to message. The note is the command remainder with
    the URL removed."""
    body = _COMMAND_RE.sub("", text or "", count=1)
    match = _URL_RE.search(body)
    if match:
        url = match.group(0)
        note = (body[: match.start()] + " " + body[match.end():])
    else:
        url = None
        for cand in entity_urls or []:
            if cand and _URL_RE.match(cand.strip()):
                url = cand.strip()
                break
        if url is None and reply_text:
            reply_match = _URL_RE.search(reply_text)
            if reply_match:
                url = reply_match.group(0)
        note = body
    note = re.sub(r"\s+", " ", note).strip()
    return url, note


def find_existing_curation(query_fn: Callable[..., list[dict]], url: str) -> dict | None:
    """Return the `hub_curations` row whose source_url normalizes to `url`'s form."""
    key = normalize_source_url(url)
    host = urlsplit(key).netloc.split(":")[0]
    if not host:
        return None
    rows = query_fn(
        "SELECT slug, title, source_url FROM hub_curations WHERE lower(source_url) LIKE %s",
        (f"%{host}%",),
    ) or []
    for row in rows:
        if normalize_source_url(row.get("source_url") or "") == key:
            return row
    return None


def find_active_curation_task(query_one_fn: Callable[..., dict | None], url_key: str) -> dict | None:
    """Return an unfinished hub_curator task already commissioned for this URL."""
    return query_one_fn(
        "SELECT id, status FROM telegram_tasks "
        "WHERE agent_type = %s AND status IN %s AND metadata->>'source_url_key' = %s "
        "ORDER BY id DESC LIMIT 1",
        (AGENT_NAME, _ACTIVE_TASK_STATUSES, url_key),
    )


def build_curation_task_content(url: str, note: str) -> str:
    lines = [
        "다음 외부 글을 읽고 /hub 큐레이션 항목을 발행하라.",
        f"URL: {url}",
        f"소유자 메모: {note.strip() or '없음'}",
        "",
        "- `source_url`은 위 URL을 글자 그대로 사용한다.",
        "- `publish_hub_curation`을 정확히 한 번 호출하면 작업이 끝난다.",
        "- 본문을 읽을 수 없을 때만 발행하지 말고, 무엇이 실패했는지 한 문단으로 보고한다.",
    ]
    return "\n".join(lines)


# ── Write-boundary validation ─────────────────────────────────────────────


def _limits() -> dict:
    from agents.hub_curator import CURATION_LIMITS

    return CURATION_LIMITS


def _hangul_ratio(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for ch in letters if _HANGUL_RE.match(ch)) / len(letters)


def _prose_problem(label: str, text: str, bounds: tuple[int, int]) -> str | None:
    lo, hi = bounds
    if "\n" in text:
        return f"{label}에 줄바꿈이 있다. 한 단락의 이어지는 문장으로 써라."
    if "**" in text or _MARKDOWN_LINK_RE.search(text) or _MARKDOWN_LINE_START_RE.match(text):
        return f"{label}에 마크다운(볼드·링크·제목·목록)이 있다. 평문으로 써라."
    if "—" in text:
        return f"{label}에 em dash(—)가 있다. 쉼표·콜론·괄호나 두 문장으로 바꿔라."
    if "북한" in text:
        return f"{label}에 '북한'이 있다. 첫 언급은 조선민주주의인민공화국, 이후 조선으로 써라."
    n = len(text)
    if n < lo:
        return f"{label}이 {n}자로 최소 {lo}자에 못 미친다. 문장을 하나 더 써라."
    if n > hi:
        return f"{label}이 {n}자로 최대 {hi}자를 넘는다. 문장을 하나 빼라."
    if _hangul_ratio(text) < 0.4:
        return f"{label}이 한국어가 아니다. 공개 텍스트는 한국어로 써라."
    return None


def validate_curation_args(args: dict, *, expected_url: str) -> str | None:
    """Return a Korean rejection reason, or None when the publish call may proceed."""
    limits = _limits()
    source_url = str(args.get("source_url") or "").strip()
    if not source_url:
        return "source_url이 비어 있다."
    if normalize_source_url(source_url) != normalize_source_url(expected_url):
        return f"source_url이 의뢰받은 URL과 다르다. 정확히 이 URL을 써라: {expected_url}"

    slug = str(args.get("slug") or "").strip().lower()
    if not slug:
        return "slug가 없다. 영어 주제어로 된 ASCII kebab-case slug를 반드시 넣어라."
    if not _SLUG_RE.match(slug) or len(slug) > int(limits["slug_max"]):
        return f"slug '{slug}'가 형식에 맞지 않는다. 소문자 a-z, 0-9, 하이픈만, 1–{limits['slug_max']}자."

    title = str(args.get("title") or "").strip()
    if not title:
        return "title이 비어 있다."
    if len(title) > int(limits["title_chars"]):
        return f"title이 {len(title)}자로 {limits['title_chars']}자를 넘는다."
    if _META_TITLE_RE.search(title):
        return "title에 메타 접두어('큐레이션', '추천', '왜 이 글이', '#N')가 있다. 글의 핵심을 말하는 평서형 헤드라인으로 써라."
    if "—" in title:
        return "title에 em dash(—)가 있다."
    if "북한" in title:
        return "title에 '북한'이 있다. 조선으로 써라."
    if _hangul_ratio(title) < 0.4:
        return "title이 한국어가 아니다."

    problem = _prose_problem(
        "selection_rationale", str(args.get("selection_rationale") or "").strip(), limits["rationale_chars"]
    )
    if problem:
        return problem
    problem = _prose_problem("context", str(args.get("context") or "").strip(), limits["context_chars"])
    if problem:
        return problem

    tags = args.get("tags")
    tags_min, tags_max = limits["tags"]
    if not isinstance(tags, list) or not (tags_min <= len(tags) <= tags_max):
        return f"tags는 {tags_min}–{tags_max}개의 짧은 한국어 태그 목록이어야 한다."
    for tag in tags:
        tag_s = str(tag or "").strip()
        if not tag_s or len(tag_s) > int(limits["tag_chars"]) or not _HANGUL_RE.search(tag_s):
            return f"tag '{tag_s}'가 규칙에 맞지 않는다. 한글 포함, {limits['tag_chars']}자 이내."

    published_at = str(args.get("source_published_at") or "").strip()
    if published_at and not _DATE_RE.match(published_at):
        return "source_published_at은 YYYY-MM-DD 형식이어야 하며, 확실하지 않으면 비워라."
    return None


def _load_task_metadata(task: dict) -> dict:
    meta = task.get("metadata") or {}
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except Exception:
            meta = {}
    return meta if isinstance(meta, dict) else {}


def make_guarded_publish_handler(inner: Callable[..., Any], task: dict) -> Callable[..., Awaitable[str]]:
    """Wrap `publish_hub_curation` for one hub_curator task."""
    expected_url = str(_load_task_metadata(task).get("source_url") or "")

    async def _guarded_publish(*args, **kwargs) -> str:
        if args:
            return ToolFailure("publish_hub_curation rejected: 인자는 이름 있는 필드로만 전달하라.")
        reason = validate_curation_args(kwargs, expected_url=expected_url)
        if reason:
            return ToolFailure(f"publish_hub_curation rejected: {reason}")
        result = inner(**kwargs)
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, str) and result.lstrip().startswith("Error:"):
            return ToolFailure(result)
        return result

    return _guarded_publish


# ── Command handler ───────────────────────────────────────────────────────


def _entity_urls(message) -> list[str]:
    urls: list[str] = []
    for source in (message, getattr(message, "reply_to_message", None)):
        if source is None:
            continue
        for entity in (getattr(source, "entities", None) or []) + (getattr(source, "caption_entities", None) or []):
            if getattr(entity, "type", "") == "text_link" and getattr(entity, "url", None):
                urls.append(entity.url)
    return urls


async def cmd_curate(message, ctx: dict) -> None:
    if not ctx["is_allowed"](message.from_user.id):
        return
    reply = getattr(message, "reply_to_message", None)
    reply_text = None
    if reply is not None:
        reply_text = getattr(reply, "text", None) or getattr(reply, "caption", None)
    url, note = parse_curate_args(getattr(message, "text", None), reply_text, _entity_urls(message))
    if not url:
        await message.answer(USAGE_TEXT)
        return

    try:
        url = await asyncio.to_thread(_validate_public_http_url, url)
    except UnsafeUrlError as exc:
        await message.answer(f"이 링크는 큐레이션할 수 없다: {exc}")
        return

    try:
        existing = await asyncio.to_thread(find_existing_curation, _query, url)
        if existing:
            await message.answer(
                f"이미 큐레이션된 링크다: {HUB_PUBLIC_URL.format(slug=existing['slug'])}\n"
                f"제목: {existing.get('title') or ''}"
            )
            return
        url_key = normalize_source_url(url)
        active = await asyncio.to_thread(find_active_curation_task, _query_one, url_key)
        if active:
            await message.answer(f"이미 진행 중인 큐레이션 태스크가 있다: #{active['id']} ({active.get('status')})")
            return

        metadata = {
            "origin": "command",
            "command": "curate",
            "source_url": url,
            "source_url_key": url_key,
            "note": note,
        }
        rows = await asyncio.to_thread(
            _query,
            "INSERT INTO telegram_tasks (user_id, content, agent_type, metadata) "
            "VALUES (%s, %s, %s, %s::jsonb) RETURNING id",
            (message.from_user.id, build_curation_task_content(url, note), AGENT_NAME, json.dumps(metadata, ensure_ascii=False)),
        )
        task_id = rows[0]["id"] if rows else None
    except Exception as exc:
        logger.error("curate command failed for %s: %s", url, exc)
        await message.answer(f"큐레이션 태스크 등록 실패: {exc}")
        return

    await message.answer(
        f"📎 큐레이션 태스크 #{task_id} 등록\n{url}\n"
        "글을 읽고 항목을 작성해 발행한 뒤 결과를 보낸다."
    )


# ── Completion hook ───────────────────────────────────────────────────────


def _failure_reason(result: dict) -> str:
    if result.get("error"):
        return str(result["error"])[:400]
    if result.get("was_interrupted"):
        return "예산 또는 라운드 한도 소진으로 발행 전에 중단됐다."
    report = str(result.get("report") or "").strip()
    if report:
        return report[:400]
    return "에이전트가 발행 없이 종료했다."


async def report_curation_outcome(
    bot,
    task: dict,
    result: dict,
    chat_id: int,
    save_system_event: Callable[[int, str, str], None] | None = None,
) -> None:
    """DM the owner what actually happened, judged by the database, not the agent."""
    meta = _load_task_metadata(task)
    source_url = str(meta.get("source_url") or "")
    row = None
    if source_url:
        try:
            row = await asyncio.to_thread(find_existing_curation, _query, source_url)
        except Exception as exc:
            logger.warning("curation outcome lookup failed for task #%s: %s", task.get("id"), exc)

    if row:
        public_url = HUB_PUBLIC_URL.format(slug=row["slug"])
        text = (
            f"✅ 큐레이션 발행 (task #{task.get('id')})\n"
            f"제목: {row.get('title') or ''}\n"
            f"{public_url}\n"
            f"slug: {row['slug']}\n"
            f"수정이 필요하면 \"큐레이션 {row['slug']} 고쳐줘\"라고 말하면 analyst가 edit_content로 처리한다."
        )
        event = f"hub curation published: {row['slug']} <- {source_url}"
    else:
        text = (
            f"❌ 큐레이션 실패 (task #{task.get('id')})\n"
            f"{source_url}\n"
            f"사유: {_failure_reason(result or {})}"
        )
        event = f"hub curation failed for {source_url}"

    try:
        await bot.send_message(chat_id=chat_id, text=text, disable_web_page_preview=not bool(row))
    except Exception as exc:
        logger.warning("curation outcome DM failed for task #%s: %s", task.get("id"), exc)
    if save_system_event is not None:
        try:
            save_system_event(chat_id, "hub_curation", event)
        except Exception as exc:
            logger.debug("curation system event save failed: %s", exc)
