"""Visitor non-publication requests for scheduled diary runs.

Web visitors sometimes ask that their own words, circumstances, affiliations
or plans stay out of the public diary. The diary agent sees recent web chats
only through a truncated preflight (a few turns per session, each cut short),
so a request made late in a long conversation, or in an earlier one, can fall
outside what it reads. Stasova's publication review never saw the chats.

This module makes the requests explicit and durable. Each visitor's messages
are read whole, once: a scan reads only what arrived after the visitor's
watermark in ``web_privacy_scans``, in chunks that never split a message, with
the visitor's standing requests in view so a later message can withdraw one.
Requests land in ``web_privacy_requests`` and stay in force until withdrawn.
The one-shot ``diary_privacy_requests`` registry call does the reading.

Before a scheduled diary run, visitors who chatted since the last diary are
scanned and every standing request is injected into the diary task as a
binding block; the save handler scans again (cheap, only new messages) and
hands the same checklist to Stasova's review and the final revision. A visitor
whose scan fails keeps its watermark, is retried next time, and is reported as
unscanned rather than clean.

Schema: ``telegram/diary_privacy.sql`` (``scripts/schema_migrations.py --only
diary-privacy``).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from llm import call_registry
from llm.json_utils import extract_json_object

logger = logging.getLogger(__name__)

FEATURE = "diary_privacy_requests"
SCHEMA_SQL = Path(__file__).with_name("diary_privacy.sql")
# Messages are batched up to this size per call; one message is never split.
CHUNK_CHARS = 60_000
MAX_VISITORS = 40
VISITOR_SQL = "COALESCE(NULLIF(fingerprint, ''), session_id)"


@dataclass
class PrivacyRequest:
    id: int
    visitor: str
    chat_log_id: int
    requested_at: str
    quote: str
    withheld: list[str]


@dataclass
class PrivacyScan:
    requests: list[PrivacyRequest] = field(default_factory=list)
    visitors: int = 0
    failed_visitors: list[str] = field(default_factory=list)


SYSTEM = (
    "너는 공개 일기를 쓰기 전에 웹 채팅 방문자의 비공개 요청을 찾는 검토자다. "
    "한 방문자의 새 메시지와, 이 방문자가 전에 한 비공개 요청 목록을 받는다. "
    "새 메시지에서 방문자가 자기 말, 신원, 개인 사정, 소속 조직, 활동, 일정 같은 정보를 일기나 공개 글에 "
    "쓰지 말라고(남기지 말라고, 빼 달라고) 요청한 경우를 찾는다. 로그가 남는지 묻는 질문처럼 요청이 아닌 우려는 제외한다.\n"
    "각 요청마다 그 요청이 막는 정보를 방문자의 메시지에서 구체적으로 풀어 적는다. "
    "\"내 이야기\"처럼 포괄적인 요청이면 방문자가 드러낸 개인·조직 정보를 항목별로 적는다. "
    "항목은 일기 작성자가 그대로 지킬 수 있게 짧고 구체적으로 쓴다(예: \"방문자의 직업과 시간 사정\", "
    "\"조직명 ○○\", \"○월 ○일 집회 일정\").\n"
    "새 메시지에서 방문자가 전에 한 요청을 거두면(이제 써도 된다 등) 그 요청 id를 revoked에 넣는다.\n"
    "반드시 JSON 객체 하나만 출력한다. 스키마: "
    "{\"requests\": [{\"message_id\": 정수, \"quote\": \"요청 원문 일부\", \"withheld\": [\"항목\", ...]}], "
    "\"revoked\": [{\"request_id\": 정수, \"message_id\": 정수}]}. "
    "해당 사항이 없으면 {\"requests\": [], \"revoked\": []}."
)


def ensure_tables() -> None:
    from db import get_conn

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SET LOCAL lock_timeout = '5s'")
        cur.execute(SCHEMA_SQL.read_text())


def _format_ts(ts) -> str:
    try:
        from shared import KST

        return ts.astimezone(KST).strftime("%Y-%m-%d %H:%M KST")
    except Exception:
        return str(ts or "?")[:19]


def _active_visitors(hours_back: int) -> list[str]:
    from db import query as db_query

    rows = db_query(
        f"SELECT {VISITOR_SQL} AS visitor, MAX(created_at) AS last_at FROM chat_logs "
        "WHERE created_at > now() - make_interval(hours => %s) AND user_query_active "
        "GROUP BY 1 ORDER BY last_at DESC LIMIT %s",
        (int(hours_back), MAX_VISITORS),
    )
    return [r["visitor"] for r in rows if r.get("visitor")]


def _new_messages(visitor: str) -> list[dict]:
    from db import query as db_query

    return db_query(
        f"SELECT c.id, c.created_at, c.user_query FROM chat_logs c "
        f"WHERE {VISITOR_SQL} = %s AND c.user_query_active AND c.id > COALESCE("
        "(SELECT last_chat_log_id FROM web_privacy_scans WHERE visitor = %s), 0) "
        "ORDER BY c.id",
        (visitor, visitor),
    )


def _standing_requests(visitor: str | None = None) -> list[PrivacyRequest]:
    from db import query as db_query

    where, params = "WHERE revoked_at IS NULL", ()
    if visitor is not None:
        where, params = where + " AND visitor = %s", (visitor,)
    rows = db_query(
        "SELECT id, visitor, chat_log_id, requested_at, quote, withheld FROM web_privacy_requests "
        f"{where} ORDER BY requested_at, id",
        params,
    )
    return [
        PrivacyRequest(
            id=r["id"], visitor=r["visitor"], chat_log_id=r["chat_log_id"],
            requested_at=_format_ts(r["requested_at"]), quote=r.get("quote") or "",
            withheld=list(r.get("withheld") or []),
        )
        for r in rows
    ]


def chunk_messages(messages: list[dict], limit: int) -> list[list[dict]]:
    chunks, current, size = [], [], 0
    for message in messages:
        length = len(str(message.get("user_query") or ""))
        if current and size + length > limit:
            chunks.append(current)
            current, size = [], 0
        current.append(message)
        size += length
    if current:
        chunks.append(current)
    return chunks


def build_prompt(messages: list[dict], standing: list[PrivacyRequest]) -> str:
    if standing:
        prior = "\n".join(
            f"[request_id={r.id}] {r.requested_at} \"{r.quote}\" — " + "; ".join(r.withheld)
            for r in standing
        )
    else:
        prior = "(없음)"
    parts = [
        f"[message_id={m['id']}] {_format_ts(m.get('created_at'))}\n{str(m.get('user_query') or '').strip()}"
        for m in messages
    ]
    return (
        f"이 방문자가 전에 한 비공개 요청:\n{prior}\n\n"
        "새 메시지 (오래된 것부터):\n\n" + "\n\n".join(parts)
    )


def parse_reply(
    reply: str | None, messages: list[dict], standing: list[PrivacyRequest],
) -> tuple[list[dict], list[tuple[int, int]]] | None:
    """(new requests, (request_id, message_id) revocations); None when unusable."""
    parsed = extract_json_object(reply or "")
    if not isinstance(parsed, dict) or not isinstance(parsed.get("requests"), list):
        return None
    by_id = {int(m["id"]): m for m in messages}
    found = []
    for item in parsed["requests"]:
        if not isinstance(item, dict):
            continue
        try:
            source = by_id.get(int(item.get("message_id")))
        except (TypeError, ValueError):
            continue
        withheld = [str(w).strip() for w in item.get("withheld") or [] if str(w).strip()]
        if source is None or not withheld:
            continue
        found.append({
            "chat_log_id": int(source["id"]),
            "requested_at": source.get("created_at"),
            "quote": str(item.get("quote") or "").strip()[:300],
            "withheld": withheld[:20],
        })
    standing_ids = {r.id for r in standing}
    revoked = []
    for item in parsed.get("revoked") or []:
        if not isinstance(item, dict):
            continue
        try:
            request_id, message_id = int(item.get("request_id")), int(item.get("message_id"))
        except (TypeError, ValueError):
            continue
        if request_id in standing_ids and message_id in by_id:
            revoked.append((request_id, message_id))
    return found, revoked


def _store(visitor: str, found: list[dict], revoked: list[tuple[int, int]], last_id: int) -> None:
    from db import get_conn

    with get_conn() as conn, conn.cursor() as cur:
        for req in found:
            cur.execute(
                "INSERT INTO web_privacy_requests (visitor, chat_log_id, requested_at, quote, withheld) "
                "VALUES (%s, %s, %s, %s, %s)",
                (visitor, req["chat_log_id"], req["requested_at"], req["quote"],
                 json.dumps(req["withheld"], ensure_ascii=False)),
            )
        for request_id, message_id in revoked:
            cur.execute(
                "UPDATE web_privacy_requests SET revoked_at = now(), revoked_by_chat_log_id = %s "
                "WHERE id = %s AND visitor = %s AND revoked_at IS NULL",
                (message_id, request_id, visitor),
            )
        cur.execute(
            "INSERT INTO web_privacy_scans (visitor, last_chat_log_id, scanned_at) VALUES (%s, %s, now()) "
            "ON CONFLICT (visitor) DO UPDATE SET last_chat_log_id = EXCLUDED.last_chat_log_id, scanned_at = now()",
            (visitor, last_id),
        )


def scan_visitor(visitor: str) -> bool:
    """Read the visitor's unread messages; False when a chunk failed (retried next time)."""
    for chunk in chunk_messages(_new_messages(visitor), CHUNK_CHARS):
        standing = _standing_requests(visitor)
        try:
            reply = call_registry.generate_sync(FEATURE, build_prompt(chunk, standing), system=SYSTEM)
        except Exception as exc:
            logger.warning("diary privacy scan failed for visitor %s: %s", visitor[:8], exc)
            return False
        parsed = parse_reply(reply, chunk, standing)
        if parsed is None:
            logger.warning("diary privacy scan: unusable reply for visitor %s", visitor[:8])
            return False
        found, revoked = parsed
        _store(visitor, found, revoked, int(chunk[-1]["id"]))
    return True


def scan(hours_back: int) -> PrivacyScan:
    """Scan visitors active in the window, then return every standing request."""
    result = PrivacyScan()
    try:
        visitors = _active_visitors(hours_back)
    except Exception as exc:
        logger.warning("diary privacy scan: visitor lookup failed: %s", exc)
        visitors = []
        result.failed_visitors.append("(visitor lookup failed)")
    result.visitors = len(visitors)
    for visitor in visitors:
        try:
            ok = scan_visitor(visitor)
        except Exception as exc:
            logger.warning("diary privacy scan failed for visitor %s: %s", visitor[:8], exc)
            ok = False
        if not ok:
            result.failed_visitors.append(visitor)
    try:
        result.requests = _standing_requests()
    except Exception as exc:
        logger.warning("diary privacy scan: standing requests unavailable: %s", exc)
        result.failed_visitors.append("(standing requests unavailable)")
    logger.info(
        "diary privacy scan: %d visitor(s), %d standing request(s), %d unscanned",
        result.visitors, len(result.requests), len(result.failed_visitors),
    )
    return result


def checklist(result: PrivacyScan) -> str:
    """Plain list shared by the diary block and the review prompts; '' when clean."""
    lines = []
    for idx, req in enumerate(result.requests, 1):
        lines.append(f"{idx}. 방문자 {req.visitor[:8]} ({req.requested_at}): \"{req.quote}\"")
        lines.extend(f"   - {item}" for item in req.withheld)
    if result.failed_visitors:
        lines.append(
            f"(방문자 {len(result.failed_visitors)}명의 메시지는 자동 검토에 실패했다. "
            "웹 채팅 로그에서 비공개 요청을 직접 확인하고, 확인하지 못한 방문자의 개인 사정·소속·일정은 옮기지 않는다.)"
        )
    return "\n".join(lines)


def format_diary_block(result: PrivacyScan) -> str:
    items = checklist(result)
    if not items:
        return ""
    return (
        "Visitor non-publication requests: binding. Visitors asked that the following stay out of public "
        "writing. Omit every listed item and any detail that would let a reader reconstruct it, even when "
        "the web-chat preflight or chat logs show it. These requests stand until the visitor withdraws them.\n"
        + items
    )
