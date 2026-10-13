"""Visitor non-publication requests for scheduled diary runs.

Web visitors sometimes ask that their own words, circumstances, affiliations
or plans stay out of the public diary. The diary agent sees recent web chats
only through a truncated preflight (a few turns per session, each cut short),
so a request made in a long conversation, or in an earlier conversation, can
fall outside what the agent reads. Stasova's publication review never saw the
chats at all.

This module makes the requests explicit. Before a scheduled diary run, every
visitor who chatted since the last diary has their own messages from the last
``LOOKBACK_DAYS`` read in full by a one-shot registry call
(``diary_privacy_requests``), which returns each request with the concrete
items it withholds. The list is injected into the diary task as a binding
block and handed to Stasova's review and the final revision as a checklist.

A visitor whose scan fails is reported as unscanned rather than silently
treated as having no request, so the agent and the reviewer know to look.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from llm import call_registry
from llm.json_utils import extract_json_object

logger = logging.getLogger(__name__)

FEATURE = "diary_privacy_requests"
# Requests are standing ("앞으로 쓰지 마라"), so look back past the diary gap.
LOOKBACK_DAYS = 30
# Newest messages are kept when a visitor's history exceeds this.
MAX_CHARS_PER_VISITOR = 120_000
MAX_VISITORS = 40

# One scan per diary task: the context builder runs it, the save handler
# reuses it for Stasova. A miss (another process, restart) scans again.
_SCANS: dict[int, "PrivacyScan"] = {}


@dataclass
class PrivacyRequest:
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
    "아래는 한 방문자가 보낸 메시지 전부다. 방문자가 자기 말, 신원, 개인 사정, 소속 조직, 활동, 일정 같은 "
    "정보를 일기나 공개 글에 쓰지 말라고(남기지 말라고, 빼 달라고) 요청한 경우만 찾는다. "
    "로그가 남는지 묻는 질문처럼 요청이 아닌 우려는 제외한다. 나중 메시지에서 철회한 요청도 제외한다.\n"
    "각 요청마다 그 요청이 막는 정보를 이 방문자의 메시지에서 구체적으로 풀어 적는다. "
    "\"내 이야기\"처럼 포괄적인 요청이면 이 방문자가 드러낸 개인·조직 정보를 항목별로 적는다. "
    "항목은 일기 작성자가 그대로 지킬 수 있게 짧고 구체적으로 쓴다(예: \"방문자의 직업과 시간 사정\", "
    "\"조직명 ○○\", \"○월 ○일 집회 일정\").\n"
    "반드시 JSON 객체 하나만 출력한다. 스키마: "
    "{\"requests\": [{\"message_id\": 정수, \"quote\": \"요청 원문 일부\", \"withheld\": [\"항목\", ...]}]}. "
    "요청이 없으면 {\"requests\": []}."
)


def _visitor_key_sql() -> str:
    return "COALESCE(NULLIF(fingerprint, ''), session_id)"


def _active_visitors(hours_back: int) -> list[str]:
    from db import query as db_query

    rows = db_query(
        f"SELECT {_visitor_key_sql()} AS visitor, MAX(created_at) AS last_at FROM chat_logs "
        "WHERE created_at > now() - make_interval(hours => %s) AND user_query_active "
        f"GROUP BY 1 ORDER BY last_at DESC LIMIT %s",
        (int(hours_back), MAX_VISITORS),
    )
    return [r["visitor"] for r in rows if r.get("visitor")]


def _visitor_messages(visitor: str) -> list[dict]:
    from db import query as db_query

    rows = db_query(
        f"SELECT id, created_at, user_query FROM chat_logs WHERE {_visitor_key_sql()} = %s "
        "AND created_at > now() - make_interval(days => %s) AND user_query_active "
        "ORDER BY id DESC",
        (visitor, LOOKBACK_DAYS),
    )
    kept, total = [], 0
    for row in rows:
        text = str(row.get("user_query") or "")
        if kept and total + len(text) > MAX_CHARS_PER_VISITOR:
            break
        kept.append(row)
        total += len(text)
    kept.reverse()
    return kept


def _format_ts(ts) -> str:
    try:
        from shared import KST

        return ts.astimezone(KST).strftime("%Y-%m-%d %H:%M KST")
    except Exception:
        return str(ts or "?")[:19]


def build_prompt(messages: list[dict]) -> str:
    parts = [
        f"[message_id={m['id']}] {_format_ts(m.get('created_at'))}\n{str(m.get('user_query') or '').strip()}"
        for m in messages
    ]
    return "방문자 메시지 (오래된 것부터):\n\n" + "\n\n".join(parts)


def parse_requests(reply: str | None, visitor: str, messages: list[dict]) -> list[PrivacyRequest] | None:
    """Requests from the model reply; None when the reply is unusable."""
    parsed = extract_json_object(reply or "")
    if not isinstance(parsed, dict) or not isinstance(parsed.get("requests"), list):
        return None
    by_id = {int(m["id"]): m for m in messages}
    found = []
    for item in parsed["requests"]:
        if not isinstance(item, dict):
            continue
        try:
            message_id = int(item.get("message_id"))
        except (TypeError, ValueError):
            continue
        source = by_id.get(message_id)
        withheld = [str(w).strip() for w in item.get("withheld") or [] if str(w).strip()]
        if source is None or not withheld:
            continue
        found.append(PrivacyRequest(
            visitor=visitor,
            chat_log_id=message_id,
            requested_at=_format_ts(source.get("created_at")),
            quote=str(item.get("quote") or "").strip()[:300],
            withheld=withheld[:20],
        ))
    return found


def scan(hours_back: int) -> PrivacyScan:
    result = PrivacyScan()
    try:
        visitors = _active_visitors(hours_back)
    except Exception as exc:
        logger.warning("diary privacy scan: visitor lookup failed: %s", exc)
        result.failed_visitors.append("(visitor lookup failed)")
        return result
    result.visitors = len(visitors)
    for visitor in visitors:
        try:
            messages = _visitor_messages(visitor)
            if not messages:
                continue
            reply = call_registry.generate_sync(FEATURE, build_prompt(messages), system=SYSTEM)
            requests = parse_requests(reply, visitor, messages)
        except Exception as exc:
            logger.warning("diary privacy scan failed for visitor %s: %s", visitor[:8], exc)
            requests = None
        if requests is None:
            result.failed_visitors.append(visitor)
        else:
            result.requests.extend(requests)
    logger.info(
        "diary privacy scan: %d visitor(s), %d request(s), %d unscanned",
        result.visitors, len(result.requests), len(result.failed_visitors),
    )
    return result


def scan_for_task(task_id: int | None, hours_back: int) -> PrivacyScan:
    if task_id is not None and task_id in _SCANS:
        return _SCANS[task_id]
    result = scan(hours_back)
    if task_id is not None:
        _SCANS.clear()  # only the running diary task matters
        _SCANS[task_id] = result
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
        "the web-chat preflight or chat logs show it. These requests apply to this entry and every later one.\n"
        + items
    )
