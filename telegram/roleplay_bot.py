"""roleplay_bot.py — Standalone DeepSeek roleplay companion (independent of Cyber-Lenin).

A lightweight second Telegram bot for free-form character roleplay. It deliberately
reuses only the verified low-level building blocks — DB pool, secret loading, the
DeepSeek client, the Anthropic-compatible tool loop, private notes, and read-only
knowledge tools — without dragging in Cyber-Lenin's agent stack (tasks, missions,
autonomous loop, KG writes, identity prompt).

Sessions live in their own tables (``roleplay_chat_history`` / ``roleplay_clear_markers``)
so this bot's conversation is fully isolated from the Cyber-Lenin Telegram bot.

Run: ``python -m telegram.roleplay_bot`` (see systemd/leninbot-roleplay.service).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from collections import Counter
from pathlib import Path

from aiogram import BaseMiddleware, Bot, Dispatcher, F, Router
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.filters import Command, CommandStart
from aiogram.types import BotCommand, CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from secrets_loader import get_secret
from db import query as _query, execute as _execute
from bot_config import _deepseek_anthropic_client, _resolve_deepseek_model
from llm.claude_loop import chat_with_tools
from llm.tool_loop_common import EMPTY_RESPONSE_FALLBACK
from runtime_tools.roleplay_jev import adjudicate_turn, PendingChoice
from runtime_tools.roleplay_actor import actor_state_view
from runtime_tools import roleplay_turn
from runtime_tools.roleplay_dynamics import with_defaults, holdout_titles, open_bargains, routine_occurrences
from runtime_tools.roleplay_pacing import policy_for, turn_time_scope
from runtime_tools.roleplay_memory import (load_notes, load_state, load_people, people_context, excluded_history_ids,
                                           get_preference, set_preference, mutate_state, set_routine)
from runtime_tools import roleplay_track
from runtime_tools.registry import TOOLS, TOOL_HANDLERS
from tool_gateway.profiles import ROLEPLAY_TELEGRAM_TOOLS
from tool_gateway.security import caller_scope, new_run_context
from tool_gateway.selection import build_toolset
from identity.prompts import EXTERNAL_SOURCE_RULE
from telegram._send_utils import make_progress_callback, split_message

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] roleplay: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────
ROLEPLAY_BOT_TOKEN = get_secret("ROLEPLAY_BOT_TOKEN", "") or ""

# Owner gate: dedicated allowlist, falling back to the main bot's allowlist.
_ALLOWED_RAW = os.getenv("ROLEPLAY_ALLOWED_USER_IDS", "") or os.getenv("ALLOWED_USER_IDS", "")
ALLOWED_USER_IDS: set[int] = {
    int(uid.strip()) for uid in _ALLOWED_RAW.split(",") if uid.strip()
}

PERSONA_PATH = Path(__file__).resolve().parent.parent / "identity" / "roleplay_persona.md"

# Roleplay tuning: flash + thinking ON for answer quality. We call DeepSeek over
# its Anthropic-compatible endpoint via claude_loop, which keeps reasoning as
# replay-only "thinking" blocks — used for quality but excluded from the
# user-facing reply (see _REPLAY_ONLY_BLOCK_TYPES in claude_loop). This is what
# separates the inner monologue from the final answer; the OpenAI path instead
# prepends reasoning to the reply, which is why it leaked.
ROLEPLAY_MODEL = _resolve_deepseek_model("deepseek_flash")  # "deepseek-v4-flash"
# Thinking and visible prose share the output allowance.
ROLEPLAY_MAX_TOKENS = int(os.getenv("ROLEPLAY_MAX_TOKENS", "16384"))
# A scene beat is usually 2–4 bookkeeping calls (time, memory, person); 8 rounds
# left no room for a single malformed call plus its retry.
ROLEPLAY_MAX_ROUNDS = int(os.getenv("ROLEPLAY_MAX_ROUNDS", "12"))
ROLEPLAY_BUDGET_USD = float(os.getenv("ROLEPLAY_BUDGET_USD", "0.50"))
HISTORY_CAP = int(os.getenv("ROLEPLAY_HISTORY_CAP", "40"))  # messages kept in context at minimum
# The window's oldest message only changes every HISTORY_STEP messages. A window
# that slid by one turn each time made the history prefix differ on every call,
# so the provider's prefix cache never covered it (25 turns at exactly the
# system+tools size). Between steps the window grows to CAP+STEP-1 messages,
# but that growth is read from cache at a fraction of the price.
HISTORY_STEP = int(os.getenv("ROLEPLAY_HISTORY_STEP", "20"))

# Curated retrieval and private notes (no task execution, no KG writes). The profile
# value lives in tool_gateway.profiles so all surface allow-lists are visible in
# one place.
_TOOL_NAMES = ROLEPLAY_TELEGRAM_TOOLS


def _select_tools() -> tuple[list[dict], dict]:
    tools, handlers = build_toolset(TOOLS, TOOL_HANDLERS, _TOOL_NAMES)
    missing = set(_TOOL_NAMES) - {str(t.get("name") or "") for t in tools}
    missing |= set(_TOOL_NAMES) - set(handlers)
    if missing:
        logger.warning("roleplay toolset missing definitions/handlers: %s", sorted(missing))
    return tools, handlers


RP_TOOLS, RP_HANDLERS = _select_tools()

# A draft whose one open Jev choice the player settles with a button. Held in
# process memory only: after a restart the player simply resends the message.
PENDING_CHOICES: dict[int, dict] = {}
FEEDBACK_PREFERENCE = "feedback_line"


# ── Persistence (own tables → session isolation) ─────────────────────
# Tables (roleplay_chat_history / roleplay_clear_markers) are created by
# scripts/schema_migrations.py ("roleplay-tables"), not at startup.
def _clear_after_id(user_id: int) -> int:
    rows = _query(
        "SELECT clear_after_id FROM roleplay_clear_markers WHERE user_id = %s",
        (user_id,),
    )
    return int(rows[0]["clear_after_id"]) if rows else 0


def save_message(user_id: int, role: str, content: str) -> None:
    _execute(
        "INSERT INTO roleplay_chat_history (user_id, role, content) VALUES (%s, %s, %s)",
        (user_id, role, content),
    )


def history_window_offset(total: int, cap: int = HISTORY_CAP, step: int = HISTORY_STEP) -> int:
    """How many of the oldest messages to skip: advances only in whole steps."""
    if total <= cap:
        return 0
    return (total - cap) // max(1, step) * max(1, step)


def load_history(user_id: int) -> list[dict]:
    """Turns after the last /new marker, oldest-first, at least HISTORY_CAP of them
    with a window start that is stable across turns (see HISTORY_STEP)."""
    min_id = _clear_after_id(user_id)
    condition = ("FROM roleplay_chat_history WHERE user_id = %s AND id > %s "
                 "AND NOT (role = 'assistant' AND content = %s) AND NOT (id = ANY(%s))")
    params = (user_id, min_id, EMPTY_RESPONSE_FALLBACK, excluded_history_ids(user_id))
    total = int(_query(f"SELECT COUNT(*) AS n {condition}", params)[0]["n"])
    rows = _query(
        f"SELECT role, content {condition} ORDER BY id ASC OFFSET %s",
        (*params, history_window_offset(total)),
    )
    return [{"role": r["role"], "content": r["content"]} for r in rows]


def reset_session(user_id: int) -> None:
    rows = _query(
        "SELECT COALESCE(MAX(id), 0) AS max_id FROM roleplay_chat_history WHERE user_id = %s",
        (user_id,),
    )
    max_id = int(rows[0]["max_id"]) if rows else 0
    _execute(
        "INSERT INTO roleplay_clear_markers (user_id, clear_after_id) VALUES (%s, %s) "
        "ON CONFLICT (user_id) DO UPDATE SET clear_after_id = EXCLUDED.clear_after_id",
        (user_id, max_id),
    )


# ── Persona / system prompt (hot-reloaded every turn) ────────────────
def build_system_prompt() -> str:
    try:
        persona = PERSONA_PATH.read_text(encoding="utf-8").strip()
    except OSError as e:
        logger.error("persona file unreadable (%s): %s", PERSONA_PATH, e)
        persona = "너는 사용자와 자유롭게 대화하는 친근한 캐릭터다. 일관된 말투를 유지한다."
    return persona + "\n\n" + EXTERNAL_SOURCE_RULE



def repeated_phrases(history: list[dict]) -> list[str]:
    """Find phrases reused across recent replies, not repetitions within one reply."""
    replies = [m["content"] for m in history if m.get("role") == "assistant"][-8:]
    counts = Counter()
    for reply in replies:
        words = re.findall(r"[\w가-힣]+", reply)
        phrases = {" ".join(words[i:i+n]) for n in range(4, 9) for i in range(len(words)-n+1)}
        counts.update(p for p in phrases if len(p) >= 12)
    selected = []
    for phrase in sorted((p for p, count in counts.items() if count >= 3), key=lambda p: (-len(p), p)):
        if not any(phrase in longer for longer in selected):
            selected.append(phrase)
        if len(selected) == 8:
            break
    return selected

# ── Telegram plumbing ────────────────────────────────────────────────
def _is_allowed(user_id: int | None) -> bool:
    return user_id is not None and user_id in ALLOWED_USER_IDS


class OwnerOnlyMiddleware(BaseMiddleware):
    """Drop messages from anyone but the configured owner(s)."""

    async def __call__(self, handler, event, data):
        user_id = getattr(getattr(event, "from_user", None), "id", None)
        if _is_allowed(user_id):
            return await handler(event, data)
        logger.info("blocked unauthorized roleplay message user_id=%s", user_id)
        return None


_split_message = split_message


def _make_progress_callback(bot: Bot, chat_id: int):
    # Stream ONLY tool steps. The model's in-character prose arrives as
    # "thinking"/"text_delta" but is also folded into the final reply by
    # the loop — streaming it here would duplicate it. Budget ("💰") is
    # mechanics noise. So a plain chat turn sends no progress at all (just
    # the reply); a tool turn shows the 🔧 steps, then the clean answer.
    progress = make_progress_callback(lambda: bot, chat_id, events=("tool_call", "tool_result"))

    async def visible_progress(event, detail):
        # Private bookkeeping must not expose the hidden state table via tool logs.
        if event in {"tool_call", "tool_result"} and re.search(r"(?:🔧 |[✓❌] )roleplay_(?:memory|state|person)(?:\(|:)", detail):
            return
        await progress(event, detail)

    visible_progress.flush = progress.flush
    return visible_progress


router = Router()


@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    await message.answer(
        "역할극 봇이 준비됐어. 그냥 말을 걸어줘.\n"
        "/new 로 대화를 처음부터 다시 시작할 수 있고, /help 로 사용법을 볼 수 있어."
    )


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(
        "이 봇은 캐릭터와 1:1 역할극을 하는 독립 봇이야 (Cyber-Lenin과 별개).\n"
        "• 그냥 메시지를 보내면 캐릭터가 답해.\n"
        "• /new — 최근 대화 초기화 (메모·상태표 유지)\n"
        "• /status — 핵심 상태, /status 상세 — 계산 근거·세부 상태. 수정·초기화는 대화로 요청해.\n"
        "• /people — 저장된 인물 목록, /people 이름 — 인물 기록 확인\n"
        "• /feedback 켜기|끄기 — 답변 뒤에 엔진이 확정한 사건·시간을 한 줄로 표시 (기본 켜짐)\n"
        "• /routine — 감옥 일과 보기, /routine 추가 06:00 아침 배식, /routine 삭제 1, /routine 비우기\n"
        "• /track 켜기|끄기 — 실존 연표(4/30 66명 조서 … 1940-02-04 처형)를 예정 사건으로 깔고 일치·이탈을 표시\n"
        "• 판정이 애매하면 버튼으로 사건을 골라 그 초안을 그대로 확정할 수 있어.\n"
        "• 캐릭터 설정은 identity/roleplay_persona.md 파일에서 편집 (재시작 불필요)\n"
        f"• 모델: {ROLEPLAY_MODEL} (thinking on, 추론은 답변에 미포함)"
    )


@router.message(Command("new"))
async def cmd_new(message: Message) -> None:
    await asyncio.to_thread(reset_session, message.from_user.id)
    await message.answer("새 대화를 시작할게. (최근 대화는 초기화했어. 저장한 메모와 상태표는 유지돼)")


@router.message(Command("feedback"))
async def cmd_feedback(message: Message) -> None:
    arg = ((message.text or "").split(maxsplit=1)[1:] or [""])[0].strip().casefold()
    if arg in {"켜기", "on", "켜"}:
        await asyncio.to_thread(set_preference, message.from_user.id, FEEDBACK_PREFERENCE, "on")
        await message.answer("확정 한 줄을 답변 뒤에 붙일게.")
    elif arg in {"끄기", "off", "꺼"}:
        await asyncio.to_thread(set_preference, message.from_user.id, FEEDBACK_PREFERENCE, "off")
        await message.answer("확정 한 줄을 숨길게. 보류된 판정의 버튼은 그대로 나와.")
    else:
        current = await asyncio.to_thread(get_preference, message.from_user.id, FEEDBACK_PREFERENCE, "on")
        await message.answer(f"확정 한 줄 표시: {'켜짐' if current != 'off' else '꺼짐'}. /feedback 켜기 또는 /feedback 끄기")


@router.message(Command("routine"))
async def cmd_routine(message: Message) -> None:
    user_id = message.from_user.id
    args = (message.text or "").split(maxsplit=2)[1:]
    verb = args[0].strip() if args else ""
    try:
        if verb in {"추가", "add"} and len(args) > 1:
            parts = args[1].split(maxsplit=1)
            if len(parts) < 2:
                raise ValueError("형식: /routine 추가 HH:MM 제목")
            time, title = parts
            state = await asyncio.to_thread(mutate_state, user_id, f"일과 추가: {time} {title}",
                lambda s: set_routine(s, s.get("routine", []) + [{"time": time, "title": title}]))
        elif verb in {"삭제", "remove"} and len(args) > 1:
            index = int(args[1].strip()) - 1
            def drop(s):
                items = list(s.get("routine", []))
                if not 0 <= index < len(items):
                    raise ValueError("그 번호의 일과가 없어")
                items.pop(index)
                return set_routine(s, items)
            state = await asyncio.to_thread(mutate_state, user_id, "일과 삭제", drop)
        elif verb in {"비우기", "clear"}:
            state = await asyncio.to_thread(mutate_state, user_id, "일과 비우기", lambda s: set_routine(s, []))
        else:
            state = await asyncio.to_thread(load_state, user_id)
    except (ValueError, TypeError) as exc:
        await message.answer(f"일과를 바꾸지 못했어: {exc}")
        return
    items = state.get("routine", [])
    lines = ["감옥 일과 (장면 시간이 이 시각을 지나면 거기서 멈추고 인물이 그 사건을 다뤄)"] if items else ["등록된 일과가 없어. /routine 추가 06:00 아침 배식"]
    lines.extend(f"{i + 1}. {item['time']} {item['title']}" for i, item in enumerate(items))
    upcoming = routine_occurrences(state, 1440)
    if upcoming:
        lines.append("다음: " + ", ".join(f"{item['title']} {at}분 뒤" for at, item in upcoming[:3]))
    elif items:
        lines.append("장면 시각이 알려져야 다음 일과를 계산할 수 있어.")
    await message.answer("\n".join(lines))


@router.message(Command("track"))
async def cmd_track(message: Message) -> None:
    user_id = message.from_user.id
    arg = ((message.text or "").split(maxsplit=1)[1:] or [""])[0].strip().casefold()
    try:
        if arg in {"켜기", "on", "켜"}:
            state = await asyncio.to_thread(mutate_state, user_id, "실존 궤도 켜기", roleplay_track.enable)
        elif arg in {"끄기", "off", "꺼"}:
            state = await asyncio.to_thread(mutate_state, user_id, "실존 궤도 끄기", roleplay_track.disable)
        else:
            state = await asyncio.to_thread(load_state, user_id)
    except ValueError as exc:
        await message.answer(f"실존 궤도를 바꾸지 못했어: {exc}")
        return
    await message.answer(_track_display(state) or "실존 궤도가 꺼져 있어. /track 켜기 — 장면 날짜 이후의 연표 사건을 예정 사건으로 깔아.")


def _track_display(state: dict) -> str:
    summary = roleplay_track.summary(state)
    if not summary:
        return ""
    parts = ["켜짐" if summary["enabled"] else "꺼짐", f"일치 {summary['matched']}", f"이탈 {summary['departed']}"]
    if summary["next"]:
        when = "지금 다룰 사건" if summary["next"]["status"] == "ready" else f"{summary['next']['days']}일 뒤"
        parts.append(f"다음: {summary['next']['title']} ({when})")
    return "실존 궤도: " + " · ".join(parts)


@router.message(Command("status"))
async def cmd_status(message: Message) -> None:
    state = dict(await asyncio.to_thread(load_state, message.from_user.id))
    detailed = (getattr(message, "text", "") or "").split()[1:] in (["상세"], ["detail"])
    people = await asyncio.to_thread(people_context, message.from_user.id, state.get("participants", []))
    names = {p["person_id"]: p["name"] for p in people.get("index", [])}
    state["participants"] = ", ".join(names.get(pid, pid) for pid in state.get("participants", [])) or "아직 지정되지 않음"
    state["saved_people"] = f"{len(names)}명 — /people로 확인"
    activities = {"rest": "휴식", "light": "가벼운 활동", "moderate": "보통 활동", "strenuous": "격한 활동", "sleep": "수면", "restrained": "억제·동결 상태", "self_care": "자기 돌봄", "focused_work": "목적 있는 작업"}
    threats = {"safe": "안전", "uncertain": "불확실", "threatening": "위협 지속", "immediate": "즉각적 위협"}
    state["activity"] = activities.get(state.get("activity"), "미설정")
    state["threat"] = threats.get(state.get("threat"), "미설정")
    state["sleep_quality"] = {"poor": "나쁨", "normal": "보통", "good": "좋음"}.get(state.get("sleep_quality"), "미설정")
    trends = {"stable": "유지", "worsening": "악화 중", "recovering": "회복 중"}
    from runtime_tools.roleplay_dynamics import injury_pain_floor
    state["pain_floor"] = f"{injury_pain_floor(state.get('injuries', [])):g}"
    from runtime_tools.roleplay_dynamics import isolation_stage
    stage = isolation_stage(state.get("isolation_minutes", 0))
    state["isolation"] = f"{state.get('isolation_minutes', 0) / 60:g}시간" + (f" — {stage['label']}: {stage['description']}" if stage else "")
    state["calm"] = f"{state.get('calm_minutes', 0) / 60:g}시간"
    state["contact_display"] = {"unknown": "미확인", "none": "교류 없음", "incidental": "배식·점검 등 짧은 접촉",
                                "hostile": "위협적 접촉", "meaningful": "지지적인 교류"}.get(state.get("social_contact", "unknown"))
    state["isolation_display"] = {"unknown": "미확인", "solitary": "강제 독방·격리", "ordinary": "일상 생활"}.get(state.get("isolation_mode", "unknown"))
    state["wakefulness"] = f"{state.get('wakefulness_minutes', 0) / 60:g}시간 상당"
    active_events = [e for e in state.get("story_events", []) if e["status"] in {"pending", "ready"}]
    event_lines = []
    for event in active_events:
        timing = "지금 다룰 사건" if event["status"] == "ready" else (
            f"{max(0, event['due_minute'] - state['scene_minute'])}분 뒤" if 'due_minute' in event else "선행 사건 뒤")
        if event.get("after_event"):
            timing += f" · {event['after_event']} 완료 조건"
        event_lines.append(f"{event['title']} ({timing})")
    state["upcoming"] = "; ".join(event_lines[:5]) + (f" 외 {len(event_lines) - 5}건" if len(event_lines) > 5 else "")
    held, lost = holdout_titles(state, "held"), holdout_titles(state, "lost")
    state["holdouts_display"] = ("; ".join(held) if held else "등록 없음") + (f" / 넘긴 것: {'; '.join(lost)}" if lost else "")
    state["bargains_display"] = "; ".join(f"{b['request']} ← {b['price']}" + (" (값 치름)" if b.get("paid") else "") for b in open_bargains(state)) or "등록 없음"
    state["routine_display"] = ", ".join(f"{item['time']} {item['title']}" for item in state.get("routine", [])) or "등록 없음"
    state["track_display"] = _track_display(state).removeprefix("실존 궤도: ") or "등록 없음"
    last_resolve = (state.get("resolve_events") or [None])[-1]
    if last_resolve:
        factors = last_resolve.get("factors", {})
        applied = [k for k in ("pain", "fatigue", "isolation", "repeat", "cap") if k in factors]
        state["resolve_event"] = (f"{factors.get('label', last_resolve['kind'])} 강도 {last_resolve['intensity']} → {last_resolve['delta']:+g} "
                                  f"({last_resolve['from']:g}→{last_resolve['to']:g})" + (f", 적용 배율: {', '.join(applied)}" if applied else ""))
    state["injuries"] = "; ".join(f"{i['description']} (심각도 {i['severity']}, {trends[i['trend']]}, {'처치함' if i['treated'] else '미처치'})" for i in state.get("injuries", [])) or "등록 없음"
    if not state.get("conditions_initialized"):
        state["injuries"] += " — 시간 계산 조건 미확인"
    from runtime_tools.roleplay_dynamics import METRICS
    for key in METRICS:
        if isinstance(state.get(key), (int, float)):
            state[key] = round(state[key], 1)
    from runtime_tools.roleplay_clock import clock_defaults
    clock = clock_defaults(state.get("clock"))
    dayparts = {"unknown": "시간대 미상", "dawn": "새벽", "morning": "아침", "afternoon": "오후", "evening": "저녁", "night": "밤"}
    day = clock["date"] or (f"{clock['year']}년 날짜 미상" if clock["year"] else "날짜 미상")
    state["calendar_display"] = f"{day} / {clock['time'] or dayparts[clock['daypart']]}"
    if clock["certainty"] == "estimated":
        state["calendar_display"] += " (추정)"
    state["relative_day"] = clock["relative_day"]
    state["time_certainty"] = {"explicit": "명시된 범위", "estimated": "추정 포함", "unknown": "미상"}[clock["certainty"]]
    interpretation = clock["last_interpretation"]
    state["time_evidence"] = f"{interpretation['source_quote']} → {interpretation['interpretation']}" if interpretation else "아직 없음"
    gaps = clock.get('uncalculated_minutes', 0)
    state["time_gaps"] = (f"{clock['unquantified_gaps']}개 구간의 활동 미상 (시간 추정 {gaps}분, 상태 미반영)" if gaps else f"{clock['unquantified_gaps']}개 구간의 경과 분량 미상 (상태 미반영)") if not clock["elapsed_complete"] else "없음"
    metrics = {"hunger": "허기", "fatigue": "피로", "pain": "통증", "tension": "긴장"}
    def display(value):
        if value is None or value == "":
            return "미설정"
        if isinstance(value, (int, float)):
            return f"{value:g}"
        return str(value)

    mental = {"resolve": "의지", "clarity": "명료함", "humiliation": "굴욕"}
    lines = ["인물 상태 (0–100)", " · ".join(f"{label}: {display(state.get(key))}" for key, label in metrics.items()),
             "정신 상태 (의지·명료함 100=굳건·또렷, 굴욕 100=극심)",
             " · ".join(f"{label}: {display(state.get(key))}" for key, label in mental.items())]
    labels = {"calendar_display": "시각", "location": "장소", "participants": "현재 장면 인물", "saved_people": "저장된 인물",
              "body": "몸 상태", "mood": "기분", "activity": "활동",
              "last_event": "직전 사건", "goal": "목적", "unresolved": "미해결", "upcoming": "예정 사건",
              "holdouts_display": "아직 지키는 것", "bargains_display": "열린 거래", "track_display": "실존 궤도"}
    if not state.get("conditions_initialized"):
        state["activity"] = "미확인"
    if not clock["elapsed_complete"]:
        labels["time_gaps"] = "시간 계산 공백"
    if detailed:
        labels.update({"avoid": "피하려는 결과", "next_action": "다음 시도",
                       "time_certainty": "시간 확실성", "relative_day": "상대 일자",
                       "time_evidence": "최근 시간 해석", "scene_minute": "장면 경과(미상 구간 추정 포함, 분)",
                       "last_calculated_minute": "마지막 계산(분)", "time_basis": "시간 근거",
                       "sleep_quality": "수면의 질", "threat": "위협 상태",
                       "injuries": "세부 부상", "pain_floor": "부상 기저 통증",
                       "isolation": "고립 누적 부담(게임 환산)", "contact_display": "교류",
                       "isolation_display": "고립 환경", "wakefulness": "각성 누적", "calm": "조용한 시간", "resolve_event": "최근 의지 사건",
                       "routine_display": "감옥 일과",
                       "reason": "최근 변경 이유"})
    lines.extend(f"{label}: {display(state.get(key))}" for key, label in labels.items()
                 if key in {"calendar_display", "location", "participants"} or state.get(key) not in (None, "", "미설정", "등록 없음"))
    if (state.get("resolve") is not None and state["resolve"] <= 25) or (state.get("humiliation") is not None and state["humiliation"] >= 75):
        lines.append("회복 경로: 작은 선택·과제 완수·경계 존중·지지 대화. 여건이 되면 자기 돌봄이나 목적 있는 작업도 가능.")
    if not detailed:
        lines.append("계산 근거·세부 부상: /status 상세")

    for chunk in split_message("\n".join(lines)):
        await message.answer(chunk)


@router.message(Command("people"))
async def cmd_people(message: Message) -> None:
    people = await asyncio.to_thread(load_people, message.from_user.id)
    args = (message.text or "").split(maxsplit=1)
    query = args[1].strip().casefold() if len(args) > 1 else ""
    if query:
        matches = [p for p in people if query in {
            p["person_id"].casefold(), p["name"].casefold(),
            *(alias.casefold() for alias in p.get("aliases", [])),
        }]
        lines = []
        if len(matches) > 1:
            lines.append("같은 이름·별칭의 기록이 여러 개 있어. 아래 식별 정보로 구분해 줘.")
        fields = {"identity": "식별 정보", "relationship": "예조프와의 관계",
                  "observed": "관찰 기록", "reported": "전해 들은 내용", "inferred": "미확정 추측"}
        for person in matches:
            lines.append(f"{person['name']} ({person['person_id']})")
            if person.get("commulingo_url"):
                lines.append("CommuLingo: " + person["commulingo_url"])
            if person.get("aliases"):
                lines.append("별칭: " + ", ".join(person["aliases"]))
            lines.extend(f"{label}: {person[key]}" for key, label in fields.items() if person.get(key))
            lines.append("")
        if not matches:
            lines = ["일치하는 인물 기록이 없어. /people에서 이름을 확인해 줘."]
    else:
        lines = [f"저장된 인물 기록: {len(people)}명"]
        for person in people:
            lines.append(f"• {person['name']} ({person['person_id']})")
            if person.get("commulingo_url"):
                lines.append(person["commulingo_url"])
        lines.append("상세 기록: /people 이름 또는 /people 식별자")
        lines.append("이 목록은 현재 장면에 없는 인물도 포함해.")
    for chunk in split_message("\n".join(lines).strip()):
        await message.answer(chunk)


@router.message(F.text)
async def handle_message(message: Message) -> None:
    user_id = message.from_user.id
    user_text = message.text or ""
    if not user_text.strip():
        return

    cached_reply = await asyncio.to_thread(roleplay_turn.committed_reply, user_id, str(message.message_id))
    if cached_reply:
        for chunk in _split_message(cached_reply):
            await message.answer(chunk)
        return
    await asyncio.to_thread(save_message, user_id, "user", user_text)
    history = await asyncio.to_thread(load_history, user_id)
    from datetime import datetime, timezone
    from llm.execution_context import attach_context, context_record
    time_policy = policy_for(user_text)
    ctx = new_run_context(interface="telegram", agent_name="roleplay", user_id=str(user_id), is_owner=True,
                          session_id=f"telegram-roleplay:{message.chat.id}", scope_type="telegram_message",
                          scope_id=str(message.message_id))
    state = None
    people = None
    try:
        state = with_defaults(await asyncio.to_thread(load_state, user_id))
        notes = await asyncio.to_thread(load_notes, user_id)
        people = await asyncio.to_thread(people_context, user_id, state.get("participants", []))
    except Exception:
        logger.exception("roleplay memory load failed")
        notes = []
    if state is None:
        await message.answer("현재 장면을 읽지 못했어. 잠시 후 다시 시도해 줘.")
        return
    try:
        authorization = await asyncio.to_thread(roleplay_turn.authorize, user_text, state, history)
    except Exception:
        logger.exception("roleplay authorization failed")
        await message.answer("이번에 진행할 장면을 확정하지 못했어. 실행할 행동이나 장면을 구체적으로 말해줘.")
        return
    time_policy = roleplay_turn.policy_for_authorization(authorization)
    history = attach_context(history, [context_record(
        "runtime_state", "telegram_roleplay_runtime", {
            "model": ROLEPLAY_MODEL, "channel": "telegram_roleplay",
            "private_notes": notes,
            # The compact view: replay bookkeeping (event IDs, timestamps) is
            # server state the model never needs and only crowds the prompt.
            "character_state": actor_state_view(state) if state else None,
            "people": people,
            "recent_repeated_phrases": repeated_phrases(history),
            "persona_time": "fictional; infer from the roleplay, not the server clock",
            "scene_direction": {"direction": roleplay_turn.direction(authorization, state)},
        }, scope=f"telegram-roleplay:{message.chat.id}",
        observed_at=datetime.now(timezone.utc), temporal_scope="current turn",
    )])

    try:
        await message.bot.send_chat_action(message.chat.id, "typing")
    except Exception:
        pass

    # Stream reasoning/tool steps as separate messages; keep the final reply clean.
    progress_cb = _make_progress_callback(message.bot, message.chat.id)
    reply = ""
    settled = None
    try:
        for attempt in range(2):
            with caller_scope(ctx), turn_time_scope(time_policy):
                with roleplay_turn.staged_memory(user_id) as stage:
                    reply = await chat_with_tools(
                        history, client=_deepseek_anthropic_client, model=ROLEPLAY_MODEL,
                        tools=RP_TOOLS, tool_handlers=RP_HANDLERS,
                        system_prompt=build_system_prompt() + "\n지금은 비공개 초안을 작성한다. 도구 저장도 검증 전 임시 기록이다. 사용자의 장면 범위를 지키고 계산 결과를 추측하지 않는다.",
                        max_rounds=ROLEPLAY_MAX_ROUNDS, max_tokens=ROLEPLAY_MAX_TOKENS,
                        continue_on_length=True, max_length_continuations=1,
                        budget_usd=ROLEPLAY_BUDGET_USD, on_progress=progress_cb,
                        agent_name="roleplay", thinking={"type":"enabled"}, output_config={"effort":"high"},
                    )
                if not reply.strip() or reply.strip() == EMPTY_RESPONSE_FALLBACK:
                    break
                try:
                    prepared = await asyncio.to_thread(roleplay_turn.prepare, user_text, state, stage['people'],
                        history, str(message.message_id), reply, authorization, stage)
                except PendingChoice as exc:
                    # Jev left one closed choice open. Ask the director instead of
                    # burning a second draft or silently dropping the turn.
                    PENDING_CHOICES[user_id] = {
                        "user_text": user_text, "state": state, "people": stage['people'], "history": history,
                        "scope_id": str(message.message_id), "draft": reply, "authorization": authorization,
                        "stage": stage, "verdict": exc.verdict, "key": exc.key,
                    }
                    await progress_cb.flush()
                    await message.answer(_choice_prompt(exc), reply_markup=_choice_keyboard(str(message.message_id), exc))
                    return
                except ValueError as exc:
                    logger.warning("roleplay draft rejected: %s", exc)
                    history = attach_context(history, [context_record('draft_revision','telegram_roleplay_runtime',
                        {'direction':'이전 초안은 폐기됐다. 범위를 지킨 새 초안을 작성하라.', 'issue':str(exc)},
                        scope=f"telegram-roleplay:{message.chat.id}",observed_at=datetime.now(timezone.utc),temporal_scope='current turn')])
                    continue
                settled = await asyncio.to_thread(adjudicate_turn, user_id, user_text, history,
                    str(message.message_id), prepared=prepared)
                if settled.get('status') in {'applied','unchanged'}:
                    reply = settled['reply']
                break
    except Exception:
        logger.exception("roleplay draft/settlement failed")
    finally:
        await progress_cb.flush()
    if settled is None or settled.get('status') not in {'applied','unchanged'}:
        await message.answer("장면을 확정하지 못해서 이번 진행은 저장하지 않았어. 어디까지 진행할지 다시 말해줘.")
        return
    await _deliver(message, user_id, reply, settled)


async def _deliver(message: Message, user_id: int, reply: str, settled: dict) -> None:
    await asyncio.to_thread(save_message, user_id, "assistant", reply)
    for chunk in _split_message(reply):
        await message.answer(chunk)
    if await asyncio.to_thread(get_preference, user_id, FEEDBACK_PREFERENCE, "on") != "off":
        state = await asyncio.to_thread(load_state, user_id)
        await message.answer(roleplay_turn.feedback_line(settled, state))


def _choice_prompt(exc: PendingChoice) -> str:
    if exc.key == "intensity":
        return "이 장면의 사건 강도를 판정하지 못했어. 골라 주면 이 초안을 그대로 확정할게."
    return "이 장면에서 실제로 일어난 사건을 판정하지 못했어. 골라 주면 이 초안을 그대로 확정할게."


def _choice_keyboard(scope_id: str, exc: PendingChoice) -> InlineKeyboardMarkup:
    rows = [[InlineKeyboardButton(text=korean, callback_data=f"rp:{scope_id}:{exc.key}:{label}")]
            for label, korean in exc.candidates]
    rows.append([InlineKeyboardButton(text="이 초안 버리기", callback_data=f"rp:{scope_id}:cancel:-")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


@router.callback_query(F.data.startswith("rp:"))
async def on_choice(query: CallbackQuery) -> None:
    user_id = query.from_user.id
    _, scope_id, key, value = (query.data or "").split(":", 3)
    pending = PENDING_CHOICES.get(user_id)
    if not pending or pending["scope_id"] != scope_id:
        await query.answer("이 선택은 더 이상 유효하지 않아. 메시지를 다시 보내줘.", show_alert=True)
        return
    if key == "cancel":
        PENDING_CHOICES.pop(user_id, None)
        await query.answer()
        await query.message.edit_text("이 초안은 버렸어. 어디까지 진행할지 다시 말해줘.")
        return
    if key != pending["key"]:
        await query.answer("이미 처리된 선택이야.")
        return
    await query.answer()
    verdict = dict(pending["verdict"])
    verdict["labels"] = {**verdict["labels"], key: value}
    chosen = next((korean for label, korean in _candidates_of(pending, key) if label == value), value)
    try:
        prepared = await asyncio.to_thread(roleplay_turn.prepare, pending["user_text"], pending["state"], pending["people"],
            pending["history"], scope_id, pending["draft"], pending["authorization"], pending["stage"], verdict)
    except PendingChoice as exc:
        pending["verdict"], pending["key"] = exc.verdict, exc.key
        await query.message.edit_text(f"선택: {chosen}\n" + _choice_prompt(exc), reply_markup=_choice_keyboard(scope_id, exc))
        return
    except ValueError as exc:
        PENDING_CHOICES.pop(user_id, None)
        logger.warning("roleplay choice completion rejected: %s", exc)
        await query.message.edit_text(f"선택: {chosen}\n그래도 초안을 확정하지 못했어 ({exc}). 다시 말해줘.")
        return
    except Exception:
        PENDING_CHOICES.pop(user_id, None)
        logger.exception("roleplay choice completion failed")
        await query.message.edit_text("확정 중 오류가 났어. 메시지를 다시 보내줘.")
        return
    PENDING_CHOICES.pop(user_id, None)
    settled = await asyncio.to_thread(adjudicate_turn, user_id, pending["user_text"], pending["history"], scope_id, prepared=prepared)
    if settled.get("status") not in {"applied", "unchanged"}:
        await query.message.edit_text(f"선택: {chosen}\n장면을 확정하지 못해서 저장하지 않았어 ({settled.get('reason')}).")
        return
    await query.message.edit_text(f"선택: {chosen}")
    await _deliver(query.message, user_id, settled["reply"], settled)


def _candidates_of(pending: dict, key: str) -> list:
    from runtime_tools.roleplay_jev import event_candidates
    if key == "intensity":
        return [("mild", "스침"), ("moderate", "보통"), ("severe", "극심")]
    return event_candidates(pending["verdict"])


async def bot_main() -> None:
    if not ROLEPLAY_BOT_TOKEN:
        raise RuntimeError("ROLEPLAY_BOT_TOKEN is not set (.env or systemd credential).")
    if not ALLOWED_USER_IDS:
        raise RuntimeError("No allowed users: set ROLEPLAY_ALLOWED_USER_IDS or ALLOWED_USER_IDS.")
    if _deepseek_anthropic_client is None:
        raise RuntimeError("DEEPSEEK_API_KEY is not configured; roleplay bot needs DeepSeek.")

    session = AiohttpSession()
    bot = Bot(token=ROLEPLAY_BOT_TOKEN, session=session)
    dp = Dispatcher()
    dp.message.middleware(OwnerOnlyMiddleware())
    dp.callback_query.middleware(OwnerOnlyMiddleware())
    dp.include_router(router)

    await bot.set_my_commands([
        BotCommand(command="new", description="새 대화 시작 (맥락 초기화)"),
        BotCommand(command="help", description="사용법"),
        BotCommand(command="status", description="인물 상태표"),
        BotCommand(command="people", description="저장된 인물 기록"),
        BotCommand(command="feedback", description="확정 한 줄 표시 켜기/끄기"),
        BotCommand(command="routine", description="감옥 일과 보기/추가/삭제"),
        BotCommand(command="track", description="실존 연표 궤도 켜기/끄기"),
    ])

    me = await bot.get_me()
    logger.info(
        "roleplay bot @%s up — model=%s tools=%s owners=%s",
        me.username, ROLEPLAY_MODEL, [t["name"] for t in RP_TOOLS], sorted(ALLOWED_USER_IDS),
    )

    try:
        await dp.start_polling(bot, drop_pending_updates=True)
    finally:
        try:
            await bot.session.close()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(bot_main())
