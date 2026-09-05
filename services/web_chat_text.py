"""Pure web-chat history, feedback, tool-trace and answer formatting."""

import json
import re
from urllib.parse import urlparse

from services.chat_history_sanitize import clean_chat_history_text
from prompt_context import uses_xml

_HISTORY_USER_CHAR_LIMIT = 6000
_HISTORY_ASSISTANT_CHAR_LIMIT = 8000
_HISTORY_TOTAL_CHAR_LIMIT = 60000


def _truncate_history_content(text: str, limit: int) -> str:
    """Keep history bounded without injecting visible system/process markers."""
    text = clean_chat_history_text(text)
    if len(text) <= limit:
        return text
    return text[-limit:].lstrip()


def _fit_history_budget(messages: list[dict], limit: int = _HISTORY_TOTAL_CHAR_LIMIT) -> list[dict]:
    """Drop the oldest history messages if per-message trimming is still too large."""
    total = sum(len(str(m.get("content", ""))) for m in messages)
    if total <= limit:
        return messages
    start = 0
    while start < len(messages) and total > limit:
        total -= len(str(messages[start].get("content", "")))
        start += 1
    return messages[start:]


_FEEDBACK_TONE_LABELS = {
    "shorter": "shorter and less digressive",
    "longer": "more developed and detailed",
    "warmer": "warmer and more emotionally responsive",
    "colder": "colder, sharper, and more severe",
    "more_direct": "more direct and less hedged",
    "more_in_character": "more strongly in character",
    "less_formal": "less formal and more conversational",
    "more_cited": "more grounded with citations or concrete references when factual",
}


def normalize_web_chat_tone_feedback(value: str | None) -> str:
    raw = str(value or "").strip().lower()
    return raw if raw in _FEEDBACK_TONE_LABELS else ""


def _render_web_tone_policy(rows: list[dict], provider: str = "claude") -> str:
    if not rows:
        return ""
    lines = [
        "Ongoing response policy inferred from the visitor's dropdown feedback. Apply as standing style policy for this persona/session; do not treat it as factual evidence and do not mention feedback history.",
    ]
    for row in rows[:3]:
        tone = normalize_web_chat_tone_feedback(row.get("tone_feedback"))
        if not tone:
            continue
        try:
            count = int(row.get("count") or 0)
        except Exception:
            count = 0
        suffix = f" (selected {count} times recently)" if count > 1 else ""
        lines.append(f"- {_FEEDBACK_TONE_LABELS[tone]}{suffix}")
    if len(lines) == 1:
        return ""
    body = "\n".join(lines)
    if uses_xml(provider):
        return f"<response-policy>\n{body}\n</response-policy>"
    return f"### Response Policy\n{body}"


def _render_web_feedback_context(rows: list[dict], provider: str = "claude") -> str:
    if not rows:
        return ""
    lines = [
        "The visitor has given manual written feedback for this next answer only. Apply it once as local style guidance, not factual evidence; do not carry it into later turns after this answer.",
    ]
    for row in rows[:8]:
        note = clean_chat_history_text(str(row.get("note") or "")).strip()[:220]
        if note:
            lines.append(f"- note={note}")
    body = "\n".join(lines)
    if uses_xml(provider):
        return f"<response-feedback>\n{body}\n</response-feedback>"
    return f"### Response Feedback\n{body}"


def _build_regeneration_message(row: dict, tone_feedback: str = "", note: str = "") -> str:
    tone_feedback = normalize_web_chat_tone_feedback(tone_feedback)
    feedback_bits: list[str] = []
    if tone_feedback:
        feedback_bits.append(_FEEDBACK_TONE_LABELS[tone_feedback])
    note = clean_chat_history_text(str(note or "")).strip()[:500]
    if note:
        feedback_bits.append(note)
    feedback = "; ".join(feedback_bits) or "Give a better alternative response while preserving the persona."
    user_query = clean_chat_history_text(str(row.get("user_query") or ""))
    previous_answer = clean_chat_history_text(str(row.get("bot_answer") or ""))[:2000]
    return (
        "Regenerate the previous answer for this same user request. "
        "Do not mention that this is a regeneration unless the character would naturally do so. "
        "Apply this feedback: " + feedback + "\n\n"
        "Original user request:\n" + user_query + "\n\n"
        "Previous answer to improve:\n" + previous_answer
    )


_SOURCE_TOOL_NAMES = {
    "knowledge_graph_search",
    "vector_search",
    "web_search",
    "fetch_url",
}

# Bound tool traces retained in recent conversation turns.
_TOOL_TRACE_ENTRY_CHAR_LIMIT = 220
_TOOL_TRACE_TOTAL_CHAR_LIMIT = 1200
_HISTORY_TOOL_TRACE_TURNS = 2


def _build_tool_trace(tool_work_details: list[str]) -> str:
    """Compress this turn's tool executions into a short text record."""
    details = [d for d in (tool_work_details or []) if str(d).strip()]
    lines: list[str] = []
    total = 0
    for detail in details:
        line = " ".join(str(detail).split())
        if len(line) > _TOOL_TRACE_ENTRY_CHAR_LIMIT:
            line = line[:_TOOL_TRACE_ENTRY_CHAR_LIMIT] + "…"
        if total + len(line) > _TOOL_TRACE_TOTAL_CHAR_LIMIT:
            lines.append(f"… (+{len(details) - len(lines)}건 생략)")
            break
        lines.append(line)
        total += len(line)
    return "\n".join(lines)

_FOOTNOTE_DEF_RE = re.compile(r"(?m)^[ \t]*\[\^(\d+)\]:[ \t]*(.*?)[ \t]*$")
_FOOTNOTE_MARKER_RE = re.compile(r"\[\^(\d+)\]")
# Parse links before bare URLs so Markdown closing parentheses never become
# part of the destination. Allow balanced parentheses inside common URLs.
_CITATION_URL = r"https?://(?:[^\s<>\"'()]|\([^\s()]*\))+"
_CITATION_RE = re.compile(
    r"\[(?P<label>[^\]]+)\]\((?P<link>" + _CITATION_URL + r")\)"
    r"|\[\^(?P<number>\d+)\]"
    r"|(?P<url>" + _CITATION_URL + r")",
    re.IGNORECASE,
)

_STORED_TARGET_RE = re.compile(
    r"(일기|게시물|공개\s*글|저장(?:된)?\s*(?:글|기록|문서|데이터)|"
    r"계정|메시지|페이지|파일|이메일|메일|diary|post|account|"
    r"stored\s+(?:content|record|document|data)|message|page|file|email)",
    re.IGNORECASE,
)
_MUTATION_ACTION_RE = re.compile(
    r"(삭제|지우|지워|비공개|수정|변경|업로드|게시|발행|전송|보내|연락|"
    r"신고|결제|송금|delete|remove|redact|unpublish|edit|change|"
    r"upload|publish|send|forward|contact|report|pay|transfer)",
    re.IGNORECASE,
)
_REQUEST_CUE_RE = re.compile(
    r"(줘|주세요|해\s*주|해라|하라|해야|맞지|않을까|"
    r"원한다|바란다|can\s+you|please|would\s+you|should\s+(?:you|we)|\?)",
    re.IGNORECASE,
)


def _normalize_source_url(value: str) -> str | None:
    candidate = str(value or "").strip().strip("<>")
    candidate = candidate.rstrip(".,;!?")
    try:
        parsed = urlparse(candidate)
    except Exception:
        return None
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return None
    return candidate


def _extract_web_source_urls(tool_work_details: list[str]) -> list[str]:
    """Return successful fetch/search result URLs, with fetched pages first."""
    fetched: list[str] = []
    searched: list[str] = []
    for raw_detail in tool_work_details or []:
        detail = str(raw_detail)
        match = _TOOL_DETAIL_RE.search(detail)
        if not match:
            continue
        tool_name = match.group(1)
        _sep, _arrow, result = detail.partition("→")
        if tool_name == "fetch_url":
            if "[fetch_url] url=" not in result:
                continue
            url_match = re.search(
                r'"url"\s*:\s*("(?:\\.|[^"\\])*")',
                detail[: match.end() + 2000],
            )
            if url_match:
                try:
                    url = _normalize_source_url(json.loads(url_match.group(1)))
                except Exception:
                    url = None
                if url:
                    fetched.append(url)
        elif tool_name == "web_search":
            if '<external source="web_search:' not in result:
                continue
            for line in result.splitlines():
                line = line.strip()
                if line.lower().startswith(("http://", "https://")):
                    url = _normalize_source_url(line)
                    if url:
                        searched.append(url)

    unique: list[str] = []
    seen: set[str] = set()
    for url in fetched + searched:
        if url not in seen:
            seen.add(url)
            unique.append(url)
    return unique


def _format_verified_url_footnotes(answer: str, source_urls: list[str]) -> str:
    """Render verified citations in first-use order, preserving link destinations."""
    candidates = list(dict.fromkeys(
        url for value in source_urls if (url := _normalize_source_url(value))
    ))
    definitions = _FOOTNOTE_DEF_RE.findall(answer)
    if not candidates and not definitions:
        # Ordinary link answers without retrieved sources are not citations.
        return answer

    body = _FOOTNOTE_DEF_RE.sub("", answer).rstrip()
    body = re.sub(
        r"(?im)\n{0,2}[ \t]*#{1,3}[ \t]*(?:출처|sources?)[ \t]*$", "", body,
    ).rstrip()
    if not candidates:
        invalid_numbers = {number for number, _ in definitions}
        return _FOOTNOTE_MARKER_RE.sub(
            lambda m: "" if m.group(1) in invalid_numbers else m.group(0), body,
        ).rstrip()

    candidate_set = set(candidates)
    old_to_url: dict[str, str] = {}
    for number, definition in definitions:
        match = _CITATION_RE.search(definition)
        url = _normalize_source_url((match.group("link") or match.group("url") or "") if match else "")
        if url in candidate_set:
            old_to_url[number] = url

    def citation_url(match: re.Match) -> str | None:
        if match.group("number"):
            return old_to_url.get(match.group("number"))
        return _normalize_source_url(match.group("link") or match.group("url"))

    # The same parser selects and renders citations; no second URL regex can
    # reinterpret an already-rendered link or lose its closing delimiter.
    chosen = list(dict.fromkeys(
        url for match in _CITATION_RE.finditer(body)
        if (url := citation_url(match)) in candidate_set
    ))
    if not chosen:
        chosen = candidates[:3]
    number_for_url = {url: number for number, url in enumerate(chosen, 1)}
    used: set[int] = set()

    def render(match: re.Match) -> str:
        url = citation_url(match)
        number = number_for_url.get(url)
        marker = f"[^{number}]" if number else ""
        if number:
            used.add(number)
        if match.group("label") is not None:
            return match.group("label") + marker
        if match.group("url"):
            # Sentence punctuation is outside the citation.
            raw = match.group("url")
            return marker + raw[len(raw.rstrip(".,;!?")):]
        return marker

    body = _CITATION_RE.sub(render, body).rstrip()
    body += "".join(f"[^{n}]" for n in range(1, len(chosen) + 1) if n not in used)
    definitions_text = "\n".join(
        f"[^{number}]: {url}" for number, url in enumerate(chosen, 1)
    )
    return f"{body}\n\n{definitions_text}".strip()


def _is_external_mutation_request(message: str) -> bool:
    compact = " ".join(str(message or "").split())
    if not _REQUEST_CUE_RE.search(compact):
        return False
    if re.search(
        r"(업로드|게시|발행|연락|신고|결제|송금|"
        r"upload|publish|contact|report|pay|transfer)",
        compact,
        re.IGNORECASE,
    ):
        return True
    return bool(_STORED_TARGET_RE.search(compact) and _MUTATION_ACTION_RE.search(compact))


def _finalize_web_answer(
    original_message: str,
    answer: str,
    tool_work_details: list[str],
) -> str:
    """Fail closed on impossible mutations, then enforce verified URL footnotes."""
    if _is_external_mutation_request(original_message):
        if re.search(r"[가-힣]", original_message):
            return (
                "이 웹 채팅은 읽기 전용이라 저장되거나 공개된 내용을 삭제·수정하거나 "
                "운영자에게 요청을 전달할 수 없다. 개인정보가 관련된 내용은 여기서 "
                "재인용하지 않으며, 권한이 있는 운영 경로에서 직접 처리해야 한다."
            )
        return (
            "This web chat is read-only. It cannot delete or edit stored/public "
            "content or forward a request to an operator. I will not repeat any "
            "personal details here; the change must be made through an authorized "
            "operator path."
        )
    source_urls = _extract_web_source_urls(tool_work_details)
    return _format_verified_url_footnotes(answer, source_urls)

_TOOL_DETAIL_RE = re.compile(r"\]\s*([A-Za-z_][A-Za-z0-9_]*)\(")


def _summarize_tool_usage(tool_work_details: list[str]) -> tuple[int, bool, str]:
    """Convert low-level tool work records into chat_logs display fields."""
    counts: dict[str, int] = {}
    source_count = 0
    for detail in tool_work_details or []:
        match = _TOOL_DETAIL_RE.search(str(detail))
        if not match:
            continue
        name = match.group(1)
        counts[name] = counts.get(name, 0) + 1
        if name in _SOURCE_TOOL_NAMES:
            source_count += 1

    web_search_used = counts.get("web_search", 0) > 0
    if not counts:
        return 0, False, ""
    strategy = "tools: " + ", ".join(
        f"{name} x{count}" for name, count in sorted(counts.items())
    )
    return source_count, web_search_used, strategy[:1000]


def _history_rows_to_messages(
    rows: list[dict], exclude_chat_log_ids: set[int] | None = None,
) -> list[dict]:
    """Render chronological rows with exclusions, deleted sides and recent traces."""
    excluded_ids = {int(x) for x in (exclude_chat_log_ids or set()) if x}
    deleted_turn_marker = "[지워진 턴]"
    if excluded_ids:
        rows = [row for row in rows if int(row.get("id") or 0) not in excluded_ids]
    # Only the most recent turns carry their tool trace: old traces add
    # tokens without teaching the model anything new about the pattern.
    trace_row_ids = {
        int(row["id"]) for row in rows[-_HISTORY_TOOL_TRACE_TURNS:]
        if row.get("id") is not None
    }
    messages = []
    for row in rows:
        if row.get("user_query"):
            messages.append({
                "role": "user",
                "content": (
                    _truncate_history_content(row["user_query"], _HISTORY_USER_CHAR_LIMIT)
                    if row.get("user_query_active", True)
                    else deleted_turn_marker
                ),
            })
        if row.get("bot_answer"):
            if row.get("bot_answer_active", True):
                content = _truncate_history_content(row["bot_answer"], _HISTORY_ASSISTANT_CHAR_LIMIT)
                trace = str(row.get("tool_trace") or "").strip()
                if trace and int(row.get("id") or 0) in trace_row_ids:
                    content = f"[도구 실행 기록]\n{trace}\n\n{content}"
            else:
                content = deleted_turn_marker
            messages.append({"role": "assistant", "content": content})
    return _fit_history_budget(messages)



def _normalize_web_result(result: str | dict, budget_tracker: dict) -> dict:
    """Give string and metadata providers the same public completion fields."""
    metadata = result if isinstance(result, dict) else {}
    return {
        "text": str(metadata.get("text") or "") if isinstance(result, dict) else str(result),
        "complete": bool(metadata.get("complete", True)),
        "truncated": bool(metadata.get("truncated", False)),
        "finish_reason": metadata.get("finish_reason"),
        "continuations_used": metadata.get("continuations_used", 0),
        "rounds": metadata.get("rounds", budget_tracker.get("rounds_used")),
        "cost_usd": metadata.get("cost_usd", budget_tracker.get("total_cost")),
    }
