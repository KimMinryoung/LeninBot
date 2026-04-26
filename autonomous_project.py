"""autonomous_project.py — Runtime for Cyber-Lenin's autonomous project loop.

Entry point: `run_tick()` — invoked hourly by `leninbot-autonomous.service`.
Picks the oldest-due active project and runs one 3-round agent wake on it.

This is the T0 pilot tier: research + planning only, no external output. See
`agents/autonomous.py` for the agent spec that enforces the tier via its tool
whitelist and prompt constraints.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv

from db import execute as db_execute, query as db_query, query_one as db_query_one
from shared import KST

load_dotenv()

logger = logging.getLogger("autonomous_project")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# ── State constants ─────────────────────────────────────────────────
STATE_RESEARCHING = "researching"
STATE_PLANNING = "planning"
STATE_PAUSED = "paused"
STATE_ARCHIVED = "archived"
ACTIVE_STATES = (STATE_RESEARCHING, STATE_PLANNING)

RECENT_NOTES_WINDOW = 8      # how many recent notes to surface in the prompt
NOTE_SNIPPET_CHARS = 500     # char cap per note when surfaced in prompt
TICK_LOG_TOOL_CAP_CHARS = 500   # char cap per tool result when persisting a tick's tool log


# ── Schema bootstrap ────────────────────────────────────────────────
_tables_ensured = False


def _ensure_tables() -> None:
    """Create autonomous_projects and autonomous_project_events if missing."""
    global _tables_ensured
    if _tables_ensured:
        return
    db_execute("""
        CREATE TABLE IF NOT EXISTS autonomous_projects (
            id            SERIAL PRIMARY KEY,
            title         TEXT NOT NULL,
            topic         TEXT NOT NULL,
            goal          TEXT NOT NULL,
            state         VARCHAR(20) NOT NULL DEFAULT 'researching',
            plan          JSONB NOT NULL DEFAULT '{"goals": [], "steps": []}'::jsonb,
            research_notes JSONB NOT NULL DEFAULT '[]'::jsonb,
            turn_count    INT NOT NULL DEFAULT 0,
            last_run_at   TIMESTAMPTZ,
            created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    db_execute("""
        CREATE TABLE IF NOT EXISTS autonomous_project_events (
            id         SERIAL PRIMARY KEY,
            project_id INT NOT NULL REFERENCES autonomous_projects(id) ON DELETE CASCADE,
            event_type VARCHAR(40) NOT NULL,
            content    TEXT,
            meta       JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    db_execute("""
        CREATE INDEX IF NOT EXISTS autonomous_project_events_project_idx
        ON autonomous_project_events(project_id, created_at DESC)
    """)
    # Operator advisories — one-shot messages that the next tick reads and then
    # the system marks consumed. Separate from research_notes (different
    # authority: operator vs agent-self) and from events (events are passive
    # logs, advisories are active directives).
    db_execute("""
        CREATE TABLE IF NOT EXISTS autonomous_project_advisories (
            id          SERIAL PRIMARY KEY,
            project_id  INT NOT NULL REFERENCES autonomous_projects(id) ON DELETE CASCADE,
            content     TEXT NOT NULL,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            consumed_at TIMESTAMPTZ
        )
    """)
    db_execute("""
        CREATE INDEX IF NOT EXISTS autonomous_project_advisories_pending_idx
        ON autonomous_project_advisories(project_id, created_at)
        WHERE consumed_at IS NULL
    """)
    _tables_ensured = True


# ── Advisories ──────────────────────────────────────────────────────
def _fetch_pending_advisories(project_id: int) -> list[dict]:
    """Return operator advisories for this project that the agent has not
    yet consumed, oldest first."""
    return db_query(
        "SELECT id, content, created_at FROM autonomous_project_advisories "
        "WHERE project_id = %s AND consumed_at IS NULL "
        "ORDER BY created_at ASC",
        (project_id,),
    )


def _mark_advisories_consumed(project_id: int, advisory_ids: list[int]) -> None:
    if not advisory_ids:
        return
    try:
        db_execute(
            "UPDATE autonomous_project_advisories SET consumed_at = NOW() "
            "WHERE project_id = %s AND id = ANY(%s)",
            (project_id, advisory_ids),
        )
    except Exception as e:
        logger.warning("Failed to mark advisories consumed (project=%s): %s", project_id, e)


def _log_event(
    project_id: int,
    event_type: str,
    content: str = "",
    meta: dict | None = None,
    *,
    max_content_chars: int = 4000,
) -> None:
    try:
        db_execute(
            "INSERT INTO autonomous_project_events(project_id, event_type, content, meta) VALUES (%s, %s, %s, %s)",
            (project_id, event_type, content[:max_content_chars], json.dumps(meta or {})),
        )
    except Exception as e:
        logger.warning("Event log failed (project=%s, type=%s): %s", project_id, event_type, e)


_TICK_TOOL_LOG_EVENT_TYPE = "tick_tool_log"


def _format_tick_tool_log(
    tool_work_details: list[str],
    *,
    per_tool_cap: int = TICK_LOG_TOOL_CAP_CHARS,
) -> str:
    """Render one tick's tool-call trace as a single bounded string.

    Each entry in `tool_work_details` already has the shape
    ``"  [round] tool_name(input_json) → full_result"``. Long tool results
    (web_search, fetch_url, kg_query) blow out arbitrary context budgets, so
    we hard-cap the trailing result portion at per_tool_cap chars. Leaves the
    header + invocation payload intact so the agent can still see WHAT it
    called and with WHAT arguments.
    """
    clipped: list[str] = []
    for entry in tool_work_details:
        s = str(entry)
        arrow = " → "
        idx = s.find(arrow)
        if idx < 0:
            clipped.append(s[: per_tool_cap * 3])  # no split; apply a coarse cap
            continue
        head = s[: idx + len(arrow)]
        tail = s[idx + len(arrow):]
        if len(tail) > per_tool_cap:
            tail = tail[: per_tool_cap - 1].rstrip() + "…"
        clipped.append(head + tail)
    return "\n".join(clipped)


def _fetch_last_tick_tool_log(project_id: int) -> dict | None:
    """Return the most recent persisted tick tool log for this project, or None."""
    try:
        rows = db_query(
            "SELECT content, meta, created_at FROM autonomous_project_events "
            "WHERE project_id = %s AND event_type = %s "
            "ORDER BY created_at DESC LIMIT 1",
            (project_id, _TICK_TOOL_LOG_EVENT_TYPE),
        )
    except Exception as e:
        logger.debug("Fetch last tick tool log failed (project=%s): %s", project_id, e)
        return None
    return rows[0] if rows else None


def _excerpt(text: str, limit: int = 500) -> str:
    """Trim a block of text to `limit` chars, collapsing whitespace and appending …."""
    if not text:
        return ""
    s = " ".join(text.split())
    return s if len(s) <= limit else s[: limit - 1].rstrip() + "…"


def _last_paragraph(text: str) -> str:
    """Return the last non-empty paragraph of `text` (agent's self-critique lives here)."""
    if not text:
        return ""
    paragraphs = [p.strip() for p in text.strip().split("\n\n") if p.strip()]
    return paragraphs[-1] if paragraphs else ""


async def _notify_telegram(project: dict, result_text: str, actions: dict, runtime: dict) -> None:
    """Send a substantive tick summary to the owner.

    Shows WHAT the agent did — note excerpts, plan rationale, state transitions —
    not just counts. Plain text, no markdown. Telegram caps messages at 4096 chars;
    we leave headroom and truncate sections as needed.
    """
    from secrets_loader import get_secret
    token = get_secret("TELEGRAM_BOT_TOKEN", "") or ""
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        logger.warning("Telegram env not set — skipping tick notification")
        return

    pid = project["id"]
    turn = (project.get("turn_count") or 0) + 1  # this tick's turn number

    parts: list[str] = [
        f"🤖 자율 프로젝트 #{pid} tick {turn} 완료",
        f"프로젝트: {project['title']}",
        f"상태: {project['state']}  /  라운드: {runtime.get('rounds_used', '?')}  /  비용: ${runtime.get('cost_usd', 0.0):.3f}",
    ]

    notes: list[str] = actions.get("notes") or []
    plan_rationale = actions.get("plan_rationale")
    state_change = actions.get("state_change")

    if not (notes or plan_rationale or state_change):
        parts.append("")
        parts.append("(도구 호출 없음 — agent가 저장 없이 종료)")
    else:
        # Show what was saved, not counts.
        for i, note_text in enumerate(notes, 1):
            prefix = f"[노트 {i}/{len(notes)}]" if len(notes) > 1 else "[노트]"
            excerpt = _excerpt(note_text, limit=900)
            parts.append("")
            parts.append(f"{prefix} {excerpt}")

        if plan_rationale:
            parts.append("")
            parts.append(f"[Plan 수정] {_excerpt(plan_rationale, limit=600)}")

        if state_change:
            frm, to, reason = state_change
            parts.append("")
            parts.append(f"[상태전환] {frm} → {to}  —  {_excerpt(reason, limit=400)}")

    # Agent's final self-critique: last paragraph of the final response.
    critique = _last_paragraph(result_text)
    if critique:
        parts.append("")
        parts.append(f"[자가비평] {_excerpt(critique, limit=500)}")

    message = "\n".join(parts)[:3900]
    try:
        from aiogram import Bot
        bot = Bot(token=token)
        try:
            await bot.send_message(chat_id=chat_id, text=message[:3800])
        finally:
            await bot.session.close()
        logger.info("Tick notification sent (project=%s, turn=%s)", pid, turn)
    except Exception as e:
        logger.warning("Telegram notify failed: %s", e)


def _collect_tick_actions(project_id: int, since_iso_utc: str) -> dict:
    """Read back events logged during this tick and extract the substantive content
    (note text, plan rationale, state transition) so the Telegram summary can show
    WHAT the agent did, not just how many times."""
    rows = db_query(
        """
        SELECT event_type, content, meta
          FROM autonomous_project_events
         WHERE project_id = %s AND created_at >= %s::timestamptz
         ORDER BY created_at ASC
        """,
        (project_id, since_iso_utc),
    )
    notes: list[str] = []
    plan_rationale: str | None = None
    state_change: tuple[str, str, str] | None = None  # (from, to, reason)
    for r in rows:
        t = r["event_type"]
        c = r.get("content") or ""
        m = r.get("meta") or {}
        if t == "note_added":
            notes.append(c)
        elif t == "plan_revised":
            plan_rationale = c
        elif t == "state_transition":
            state_change = (str(m.get("from")), str(m.get("to")), c)
    return {"notes": notes, "plan_rationale": plan_rationale, "state_change": state_change}


# ── Project selection ───────────────────────────────────────────────
def _pick_next_project() -> dict | None:
    """Return the active project with the oldest last_run_at (NULLS FIRST)."""
    _ensure_tables()
    return db_query_one(
        """
        SELECT id, title, topic, goal, state, plan, research_notes, turn_count,
               last_run_at, created_at
          FROM autonomous_projects
         WHERE state = ANY(%s)
         ORDER BY last_run_at ASC NULLS FIRST, id ASC
         LIMIT 1
        """,
        (list(ACTIVE_STATES),),
    )


# ── Custom tool definitions (project-scoped) ────────────────────────
#
# These tools close over the current project_id so the agent can't accidentally
# mutate a different project. They are registered fresh on every tick.

def _build_project_tools(project_id: int) -> tuple[list[dict], dict]:
    """Return (tool_schemas, handlers) for add_research_note / revise_plan / set_project_state."""

    async def _handle_add_note(text: str = "", sources: list | None = None) -> str:
        text = (text or "").strip()
        if not text:
            return "error: text is required"
        if sources is None:
            sources = []
        elif not isinstance(sources, list):
            sources = [str(sources)]
        note = {
            "turn": _current_turn_counter(project_id),
            "text": text[:6000],
            "sources": [str(s)[:500] for s in sources][:20],
            "created_at": datetime.now(KST).isoformat(timespec="seconds"),
        }
        db_execute(
            """
            UPDATE autonomous_projects
               SET research_notes = research_notes || %s::jsonb,
                   updated_at = NOW()
             WHERE id = %s
            """,
            (json.dumps([note]), project_id),
        )
        _log_event(project_id, "note_added", text[:400], {"sources": note["sources"]})
        return f"ok: note saved ({len(note['sources'])} sources)"

    async def _handle_revise_plan(
        rationale: str = "",
        goals: list | None = None,
        steps: list | None = None,
    ) -> str:
        rationale = (rationale or "").strip()
        if not rationale:
            return "error: rationale is required"
        goals = goals or []
        steps = steps or []
        if not isinstance(goals, list) or not isinstance(steps, list):
            return "error: goals and steps must be lists of strings"
        new_plan = {
            "goals": [str(g)[:500] for g in goals][:20],
            "steps": [str(s)[:500] for s in steps][:40],
            "revised_at": datetime.now(KST).isoformat(timespec="seconds"),
            "rationale": rationale[:1000],
        }
        # Preserve old plan in the event log before overwriting.
        old = db_query_one("SELECT plan FROM autonomous_projects WHERE id = %s", (project_id,))
        db_execute(
            "UPDATE autonomous_projects SET plan = %s, updated_at = NOW() WHERE id = %s",
            (json.dumps(new_plan), project_id),
        )
        _log_event(
            project_id, "plan_revised", rationale[:400],
            {"previous_plan": (old or {}).get("plan"), "new_plan": new_plan},
        )
        return f"ok: plan revised ({len(new_plan['goals'])} goals, {len(new_plan['steps'])} steps)"

    async def _handle_set_state(state: str = "", reason: str = "") -> str:
        target = (state or "").strip().lower()
        reason = (reason or "").strip()
        allowed = {STATE_RESEARCHING, STATE_PLANNING, STATE_PAUSED, STATE_ARCHIVED}
        if target not in allowed:
            return f"error: state must be one of {sorted(allowed)}"
        if not reason:
            return "error: reason is required"
        current = db_query_one("SELECT state FROM autonomous_projects WHERE id = %s", (project_id,))
        if current and current["state"] == target:
            return f"noop: already in state {target}"
        db_execute(
            "UPDATE autonomous_projects SET state = %s, updated_at = NOW() WHERE id = %s",
            (target, project_id),
        )
        _log_event(
            project_id, "state_transition", reason[:400],
            {"from": (current or {}).get("state"), "to": target},
        )
        return f"ok: state → {target}"

    schemas = [
        {
            "name": "add_research_note",
            "description": (
                "Persist a single research finding to the project's note log. Use this IMMEDIATELY after "
                "a research step — chat memory does not persist across hourly ticks, only notes do."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The finding, in dense prose. Cite sources inline."},
                    "sources": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "URLs, KG node ids, or vector-DB ids that back this finding. Unsourced notes are discouraged.",
                    },
                },
                "required": ["text"],
            },
        },
        {
            "name": "revise_plan",
            "description": (
                "Overwrite the project's plan. The previous plan is preserved in the event log. "
                "Use only when accumulated research genuinely justifies a change — not every tick."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "goals": {"type": "array", "items": {"type": "string"}, "description": "Top-level outcomes the project pursues."},
                    "steps": {"type": "array", "items": {"type": "string"}, "description": "Ordered concrete steps to advance the goals."},
                    "rationale": {"type": "string", "description": "Why the previous plan was insufficient. Required."},
                },
                "required": ["rationale"],
            },
        },
        {
            "name": "set_project_state",
            "description": (
                "Transition the project state. Allowed: researching | planning | paused | archived. "
                "Do not flip state every tick; justify transitions in `reason`."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "state": {"type": "string", "enum": ["researching", "planning", "paused", "archived"]},
                    "reason": {"type": "string", "description": "Why this transition is warranted. Required."},
                },
                "required": ["state", "reason"],
            },
        },
    ]
    handlers = {
        "add_research_note": _handle_add_note,
        "revise_plan": _handle_revise_plan,
        "set_project_state": _handle_set_state,
    }
    return schemas, handlers


def _current_turn_counter(project_id: int) -> int:
    row = db_query_one("SELECT turn_count FROM autonomous_projects WHERE id = %s", (project_id,))
    return int(row["turn_count"]) if row else 0


# ── Prompt context assembly ─────────────────────────────────────────
def _recent_notes(project: dict) -> list[dict]:
    notes = project.get("research_notes") or []
    if isinstance(notes, str):
        try:
            notes = json.loads(notes)
        except Exception:
            notes = []
    return notes[-RECENT_NOTES_WINDOW:]


def _format_plan(plan) -> str:
    if isinstance(plan, str):
        try:
            plan = json.loads(plan)
        except Exception:
            return "(plan parse error)"
    if not plan:
        return "(empty — no plan yet)"
    goals = plan.get("goals") or []
    steps = plan.get("steps") or []
    if not goals and not steps:
        return "(empty — no plan yet)"
    out = []
    if goals:
        out.append("Goals:")
        out.extend(f"  - {g}" for g in goals)
    if steps:
        out.append("Steps:")
        out.extend(f"  {i+1}. {s}" for i, s in enumerate(steps))
    revised = plan.get("revised_at")
    if revised:
        out.append(f"(Last revised: {revised})")
    return "\n".join(out)


def _format_notes(notes: list[dict]) -> str:
    if not notes:
        return "(no prior notes — this is an early tick)"
    lines = []
    for n in notes:
        snippet = (n.get("text") or "")[:NOTE_SNIPPET_CHARS]
        sources = n.get("sources") or []
        src_str = f" [{', '.join(sources[:3])}{'…' if len(sources) > 3 else ''}]" if sources else ""
        lines.append(f"- (turn {n.get('turn', '?')}, {n.get('created_at', '?')}) {snippet}{src_str}")
    return "\n".join(lines)


def _uses_xml_prompt(provider: str | None) -> bool:
    return (provider or "claude") == "claude"


def _build_task_prompt(
    project: dict,
    turn_budget: int,
    advisories: list[dict] | None = None,
    *,
    provider: str = "claude",
) -> str:
    plan_text = _format_plan(project.get("plan"))
    notes_text = _format_notes(_recent_notes(project))
    if not _uses_xml_prompt(provider):
        parts = [
            f"### Project\n- **id**: {project['id']}\n- **title**: {project['title']}\n- **topic**: {project['topic']}",
            f"### Goal\n\n{project['goal']}",
        ]
        if advisories:
            advisory_lines = []
            for a in advisories:
                ts = a["created_at"].astimezone(KST).strftime("%Y-%m-%d %H:%M KST") if a.get("created_at") else "?"
                advisory_lines.append(f"#### Operator advice @ {ts}\n\n{a['content']}")
            parts.append(
                "### Operator Advice\n\n"
                "Read these messages before acting. They override your prior plan when they conflict. "
                "These messages are shown once; after this tick they are marked consumed.\n\n"
                + "\n\n".join(advisory_lines)
            )
        parts.extend([
            f"### State\n\n{project['state']}",
            f"### Plan\n\n{plan_text}",
            f"### Recent Notes\n\n{notes_text}",
        ])
        last_log = _fetch_last_tick_tool_log(project["id"])
        if last_log and last_log.get("content"):
            meta = last_log.get("meta") or {}
            prior_turn = meta.get("turn", "?")
            prior_cost = meta.get("cost_usd")
            prior_rounds = meta.get("rounds_used", "?")
            header_bits = [f"turn={prior_turn}", f"rounds={prior_rounds}"]
            if prior_cost is not None:
                header_bits.append(f"cost=${prior_cost:.3f}")
            parts.append(
                f"### Last Tick Execution ({', '.join(header_bits)})\n\n"
                "Raw tool trace from your previous tick. Call/arguments are shown in full; "
                "results are clipped at 500 chars each. Do not re-run queries that already "
                "yielded what you need.\n\n"
                f"```text\n{last_log['content']}\n```"
            )
        parts.extend([
            f"### Turn Budget\n\n{turn_budget} rounds this tick. Use them deliberately.",
            f"### Tick Info\n\nturn_count so far: {project.get('turn_count', 0)}; "
            f"last_run_at: {project.get('last_run_at') or 'never'}",
            "",
            "Advance this project by exactly one concrete step. Save findings via add_research_note "
            "BEFORE your final response. End with a one-paragraph self-critique: did this tick "
            "advance the goal? what is the next tick's focus?",
        ])
        return "\n\n".join(parts)

    parts = [
        f"<project>\nid: {project['id']}\ntitle: {project['title']}\ntopic: {project['topic']}\n</project>",
        f"<goal>\n{project['goal']}\n</goal>",
    ]
    # Operator advisories sit between <goal> and <state> — above the plan so
    # they can redirect it. Omitted entirely when empty.
    if advisories:
        advisory_lines = []
        for a in advisories:
            ts = a["created_at"].astimezone(KST).strftime("%Y-%m-%d %H:%M KST") if a.get("created_at") else "?"
            advisory_lines.append(f"[from operator @ {ts}]\n{a['content']}")
        parts.append(
            "<operator-advice>\n"
            "The following messages were left for you by the operator between your last tick "
            "and this one. Read them BEFORE acting. They override your prior plan when they "
            "conflict — the operator sees context you don't. These messages are shown once; "
            "after this tick they are marked consumed.\n\n"
            + "\n\n".join(advisory_lines)
            + "\n</operator-advice>"
        )
    parts.extend([
        f"<state>{project['state']}</state>",
        f"<plan>\n{plan_text}\n</plan>",
        f"<recent-notes>\n{notes_text}\n</recent-notes>",
    ])

    # Previous tick's full tool trace — what YOU called and what came back.
    # research_notes capture curated findings; this captures raw execution so
    # the agent can remember "I already ran web_search on X, got Y" without
    # having to re-issue the call. Bounded by per-tool 500-char cap.
    last_log = _fetch_last_tick_tool_log(project["id"])
    if last_log and last_log.get("content"):
        meta = last_log.get("meta") or {}
        prior_turn = meta.get("turn", "?")
        prior_cost = meta.get("cost_usd")
        prior_rounds = meta.get("rounds_used", "?")
        header_bits = [f"turn={prior_turn}", f"rounds={prior_rounds}"]
        if prior_cost is not None:
            header_bits.append(f"cost=${prior_cost:.3f}")
        parts.append(
            f"<last-tick-execution {' '.join(header_bits)}>\n"
            f"Raw tool trace from your previous tick — call/arguments shown in full, "
            f"results clipped at 500 chars each. Do NOT re-run these queries if they "
            f"already yielded what you need; build on them instead.\n\n"
            f"{last_log['content']}\n"
            f"</last-tick-execution>"
        )

    parts.extend([
        f"<turn-budget>{turn_budget} rounds this tick. Use them deliberately.</turn-budget>",
        f"<tick-info>turn_count so far: {project.get('turn_count', 0)}; "
        f"last_run_at: {project.get('last_run_at') or 'never'}</tick-info>",
        "",
        "Advance this project by exactly one concrete step. Save findings via add_research_note "
        "BEFORE your final response. End with a one-paragraph self-critique: did this tick "
        "advance the goal? what is the next tick's focus?",
    ])
    return "\n\n".join(parts)


# ── Tick execution ──────────────────────────────────────────────────
async def _run_one_tick(project: dict) -> dict:
    """Run a single agent wake on the given project. Returns a result dict."""
    from agents import get_agent
    from claude_loop import dedupe_tools_by_name
    from bot_config import (
        _get_autonomous_provider, _get_model_by_alias,
        get_current_model_selection,
    )
    import telegram.tools as tt_module

    spec = get_agent("autonomous_project")
    configured_provider = _get_autonomous_provider()
    provider = spec.effective_provider(configured_provider)

    # Compose tool set: spec-filtered base tools + project-scoped custom tools.
    base_tools = tt_module.TOOLS
    base_handlers = tt_module.TOOL_HANDLERS
    agent_tools, agent_handlers = spec.filter_tools(base_tools, base_handlers)

    project_tools, project_handlers = _build_project_tools(project["id"])
    agent_tools.extend(project_tools)
    agent_handlers.update(project_handlers)
    agent_tools = dedupe_tools_by_name(agent_tools)

    # Fully static spec prompt — cacheable. Current time is injected as runtime
    # context into the user message below so the system prompt never drifts.
    system_prompt = spec.render_prompt(provider=provider)

    # Fetch pending operator advisories. Marked consumed AFTER the tick
    # succeeds — if the tick raises, advisories remain pending for the next
    # tick to see so the operator's message isn't lost to an infra error.
    pending_advisories = _fetch_pending_advisories(project["id"])
    user_content = _build_task_prompt(
        project,
        turn_budget=spec.max_rounds,
        advisories=pending_advisories,
        provider=provider,
    )

    # Autonomous uses its own model tier, independent from chat/task settings.
    selected = get_current_model_selection("autonomous", provider_override=provider)
    model = selected.get("model_id") or None
    if provider == "claude":
        model = await _get_model_by_alias(str(selected.get("alias") or "sonnet"))
    model_for_log = model or "unknown"
    budget_tracker: dict = {}

    # Use tz-aware UTC so Postgres compares against TIMESTAMPTZ unambiguously
    # regardless of the DB session's timezone setting.
    tick_started_at_utc = datetime.now(timezone.utc).isoformat()
    _log_event(
        project["id"], "tick_start",
        f"turn #{(project.get('turn_count') or 0) + 1}, state={project['state']}",
        {"provider": provider, "model": model_for_log,
         "max_rounds": spec.max_rounds, "budget_usd": spec.budget_usd},
    )

    tick_messages = [{"role": "user", "content": user_content}]

    try:
        from telegram.bot import _chat_with_tools

        result_text = await _chat_with_tools(
            tick_messages,
            model=model,
            system_prompt=system_prompt,
            max_rounds=spec.max_rounds,
            max_tokens=16384,  # was 4096 — too small for long-form drafts; agent hit
                              # truncation mid-note and never reached the final
                              # add_research_note call (silent tick-0-save failure)
            budget_usd=spec.budget_usd,
            extra_tools=agent_tools,
            extra_handlers=agent_handlers,
            budget_tracker=budget_tracker,
            provider_override=provider,
            agent_name="autonomous_project",
            runtime_kind="autonomous",
            finalization_tools=spec.finalization_tools,
            terminal_tools=spec.terminal_tools,
        )
    except Exception as e:
        logger.exception("Tick failed for project %s", project["id"])
        _log_event(project["id"], "tick_error", str(e)[:2000])
        raise

    # Increment turn counter and last_run_at AFTER the agent loop completes.
    db_execute(
        "UPDATE autonomous_projects SET turn_count = turn_count + 1, last_run_at = NOW(), updated_at = NOW() WHERE id = %s",
        (project["id"],),
    )
    # Mark pending advisories as consumed — tick saw them, job done.
    if pending_advisories:
        _mark_advisories_consumed(
            project["id"], [a["id"] for a in pending_advisories]
        )
        _log_event(
            project["id"], "advisories_consumed",
            f"{len(pending_advisories)} advisories marked consumed",
            {"ids": [a["id"] for a in pending_advisories]},
        )
    _log_event(
        project["id"], "tick_end",
        (result_text or "")[:3000],
        {"cost_usd": round(budget_tracker.get("total_cost", 0.0), 4),
         "rounds_used": budget_tracker.get("rounds_used", 0)},
    )

    # Persist this tick's tool-call trace so the next tick can see WHAT this
    # tick actually ran and WHAT came back. Curated research_notes only capture
    # final findings; the raw tool log captures execution (retries, dead-end
    # searches, confirmed-negative results) that would otherwise be lost.
    _raw_tool_details = budget_tracker.get("tool_work_details") or []
    if _raw_tool_details:
        _tick_log_str = _format_tick_tool_log(_raw_tool_details)
        _log_event(
            project["id"], _TICK_TOOL_LOG_EVENT_TYPE,
            _tick_log_str,
            {"turn": (project.get("turn_count") or 0) + 1,
             "cost_usd": round(budget_tracker.get("total_cost", 0.0), 4),
             "rounds_used": budget_tracker.get("rounds_used", 0),
             "tool_calls": len(_raw_tool_details)},
            # Tool logs are the agent's working memory between ticks — don't
            # clip at the 4KB default that's fine for human-readable events.
            max_content_chars=40000,
        )

    # Telegram notification — runs after tick data is committed so a notify
    # failure can never lose work. _notify_telegram swallows its own errors.
    actions = _collect_tick_actions(project["id"], tick_started_at_utc)
    runtime = {
        "cost_usd": budget_tracker.get("total_cost", 0.0),
        "rounds_used": budget_tracker.get("rounds_used", 0),
    }
    await _notify_telegram(project, result_text or "", actions, runtime)

    return {
        "project_id": project["id"],
        "result_text": result_text or "",
        "cost_usd": budget_tracker.get("total_cost", 0.0),
        "rounds_used": budget_tracker.get("rounds_used", 0),
    }


def run_tick() -> dict | None:
    """Synchronous entry point for systemd. Picks the next project and advances it."""
    from bot_config import is_autonomous_active
    if not is_autonomous_active():
        logger.info("Autonomous loop paused via config (autonomous_active=false) — skipping tick.")
        return None
    _ensure_tables()
    project = _pick_next_project()
    if not project:
        logger.info("No active projects — nothing to do.")
        return None
    logger.info(
        "Advancing project #%s %r (state=%s, turn=%s)",
        project["id"], project["title"], project["state"], project.get("turn_count"),
    )
    result = asyncio.run(_run_one_tick(project))
    logger.info(
        "Tick complete: project=%s cost=$%.4f rounds=%s",
        result["project_id"], result["cost_usd"], result["rounds_used"],
    )
    return result


if __name__ == "__main__":
    # Direct invocation for manual/systemd use: `python -m autonomous_project`
    # or `python autonomous_project.py`.
    try:
        run_tick()
    except Exception:
        logger.exception("run_tick failed")
        sys.exit(1)
