"""telegram_tasks.py — Background task processing, scheduling, and system monitoring.

Extracted from telegram_bot.py for modularity.
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timezone

from aiogram import Bot
from aiogram.types import BufferedInputFile
from db import query as _query, execute as _execute, query_one as _query_one

from shared import KST

logger = logging.getLogger(__name__)
_SCRATCHPAD_MAX_CHARS = 20_000
_DEFAULT_MAX_RESUME_ATTEMPTS = 2
_STARTUP_HANDOFF_MARKER = "## Checkpoint: startup handoff"
_MAX_TASK_CHAIN_DEPTH = 5
_SHUTDOWN_CHECKPOINT_MARKER = "## Checkpoint: shutdown before restart"
_RESTART_COMPLETED_MARKER = "[restart already completed by parent task]"


# ── Current State Builder (shared by orchestrator + task agents) ─────

def build_current_state(user_id: int, *, detail_level: str = "high") -> str:
    """Build a <current_state> block showing completed/in-progress/pending tasks.

    Args:
        user_id: Telegram user ID
        detail_level: "high" = orchestrator (summaries only), "low" = brief
    Returns:
        XML string or empty string if no relevant state
    """
    try:
        now_ts = datetime.now(KST).strftime("%Y-%m-%dT%H:%M+09:00")

        # Active mission
        mission_line = ""
        try:
            mission_rows = _query(
                "SELECT id, title FROM telegram_missions WHERE user_id = %s AND status = 'active' "
                "ORDER BY created_at DESC LIMIT 1",
                (user_id,),
            )
            if mission_rows:
                m = mission_rows[0]
                mission_line = f'  <active_mission id="{m["id"]}">{m["title"]}</active_mission>\n'
        except Exception:
            pass

        # Completed tasks (last 24h)
        done_rows = _query(
            "SELECT id, agent_type, content, result, completed_at FROM telegram_tasks "
            "WHERE user_id = %s AND status = 'done' AND completed_at > NOW() - INTERVAL '24 hours' "
            "ORDER BY completed_at DESC LIMIT 5",
            (user_id,),
        )
        done_rows.reverse()

        # In-progress tasks
        processing_rows = _query(
            "SELECT id, agent_type, content, created_at FROM telegram_tasks "
            "WHERE user_id = %s AND status = 'processing' "
            "ORDER BY created_at ASC",
            (user_id,),
        )

        # Pending tasks
        pending_rows = _query(
            "SELECT id, agent_type, content, created_at FROM telegram_tasks "
            "WHERE user_id = %s AND status = 'pending' "
            "ORDER BY created_at ASC LIMIT 5",
            (user_id,),
        )

        if not done_rows and not processing_rows and not pending_rows and not mission_line:
            return ""

        lines = [f'<current_state timestamp="{now_ts}">']
        if mission_line:
            lines.append(mission_line.rstrip())

        # Completed
        if done_rows:
            lines.append("  <completed>")
            for t in done_rows:
                agent = t.get("agent_type") or "analyst"
                result_summary = (str(t.get("result") or "")[:150]).replace("\n", " ").strip()
                if not result_summary:
                    result_summary = (str(t.get("content") or "")[:80]).replace("\n", " ")
                ts = str(t.get("completed_at") or "")[:16]
                lines.append(f"    - [{agent}] #{t['id']} ({ts}): {result_summary}")
            lines.append("  </completed>")

        # In-progress
        if processing_rows:
            lines.append("  <in_progress>")
            for t in processing_rows:
                agent = t.get("agent_type") or "analyst"
                content_brief = (str(t.get("content") or "")[:100]).replace("\n", " ")
                lines.append(f"    - [{agent}] #{t['id']}: {content_brief}")
            lines.append("  </in_progress>")

        # Pending
        if pending_rows:
            lines.append("  <not_started>")
            for t in pending_rows:
                agent = t.get("agent_type") or "analyst"
                content_brief = (str(t.get("content") or "")[:100]).replace("\n", " ")
                lines.append(f"    - [{agent}] #{t['id']}: {content_brief}")
            lines.append("  </not_started>")

        lines.append("</current_state>")
        return "\n".join(lines)

    except Exception as e:
        logger.debug("build_current_state failed: %s", e)
        return ""


# ── Task Report Helpers ──────────────────────────────────────────────

def _extract_summary(report: str, max_len: int = 300) -> str:
    """Extract Executive Summary section or first paragraph as brief summary."""
    for marker in ("## Executive Summary", "## 요약", "## 핵심 요약"):
        idx = report.find(marker)
        if idx != -1:
            after = report[idx + len(marker):].strip()
            next_heading = after.find("\n## ")
            section = after[:next_heading].strip() if next_heading != -1 else after
            if section:
                return section[:max_len] + ("..." if len(section) > max_len else "")
    for line in report.split("\n"):
        line = line.strip()
        if line and not line.startswith("#") and not line.startswith("**"):
            return line[:max_len] + ("..." if len(line) > max_len else "")
    return report[:max_len]


def _classify_priority(content: str, report: str) -> str:
    """Classify task result priority from content tags or report urgency keywords."""
    if "[🔴 HIGH]" in content:
        return "high"
    if "[🟢 LOW]" in content:
        return "low"
    report_lower = report[:2000].lower()
    if any(k in report_lower for k in ("urgent", "critical", "긴급", "위기", "경고", "즉시")):
        return "high"
    return "normal"


def _append_task_scratchpad(task_id: int, note: str) -> None:
    """Append a checkpoint note to telegram_tasks.scratchpad."""
    rows = _query("SELECT scratchpad FROM telegram_tasks WHERE id = %s", (task_id,))
    current = (rows[0].get("scratchpad") or "") if rows else ""
    new_pad = f"{current}\n{note}".strip() if current else note
    if len(new_pad) > _SCRATCHPAD_MAX_CHARS:
        new_pad = new_pad[-_SCRATCHPAD_MAX_CHARS:]
    _execute("UPDATE telegram_tasks SET scratchpad = %s WHERE id = %s", (new_pad, task_id))


# ── Process Task ─────────────────────────────────────────────────────

async def process_task(
    bot: Bot,
    task: dict,
    *,
    chat_with_tools_fn,
    get_model_fn,
    task_system_prompt: str,
    max_tokens_task: int,
    allowed_user_ids: set[int],
    log_event_fn,
    extra_tools: list | None = None,
    extra_handlers: dict | None = None,
    budget_usd: float = 1.00,
    on_progress=None,
    on_complete=None,
):
    """Process a task: run tools, generate report, save to DB, send as file.

    Args:
        bot: Telegram Bot instance.
        task: Dict with id, user_id, content, parent_task_id, depth.
        chat_with_tools_fn: Async callable matching _chat_with_tools signature.
        get_model_fn: Callable returning current model ID.
        task_system_prompt: System prompt for task execution.
        max_tokens_task: Max tokens for task output.
        allowed_user_ids: Set of allowed Telegram user IDs.
        log_event_fn: Callable for persistent error logging.
        extra_tools: Additional tool definitions (e.g. task-context tools).
        extra_handlers: Additional tool handlers.
        budget_usd: USD budget for this task (default $1.00).
        on_progress: Optional async callback for live progress updates.
        on_complete: Optional callback(task_id, status, summary) called after
            task finishes. Used to notify the orchestrator of completion.
    """
    task_id = task["id"]
    user_id = task["user_id"]
    content = task["content"]
    scratchpad = task.get("scratchpad") or ""
    depth = task.get("depth") or 0
    parent_task_id = task.get("parent_task_id")
    mission_id = task.get("mission_id")
    is_self_generated = (user_id == 0)

    # Inject mission context using task's own mission_id
    mission_ctx = ""
    if mission_id:
        try:
            from telegram_mission import get_mission_events, add_mission_event
            # Look up mission title
            from db import query as _db_query
            mission_rows = _db_query("SELECT title FROM telegram_missions WHERE id = %s", (mission_id,))
            mission_title = mission_rows[0]["title"] if mission_rows else "?"
            events = get_mission_events(mission_id, limit=20)
            if events:
                lines = [f"<mission-context id=\"{mission_id}\" title=\"{mission_title}\">"]
                for e in events:
                    lines.append(f"  [{e['created_at']}] ({e['source']}) {e['event_type']}: {str(e['content'] or '')[:500]}")
                lines.append("</mission-context>")
                mission_ctx = "\n".join(lines)
            try:
                from bot_config import get_current_model_selection
                model_sel = get_current_model_selection("task")
                mission_ctx = (
                    mission_ctx + "\n" if mission_ctx else ""
                ) + (
                    f"<runtime-model-context>\n"
                    f"  <current-model provider=\"{model_sel['provider']}\" tier=\"{model_sel['tier']}\" alias=\"{model_sel['alias']}\">{model_sel['model_id']}</current-model>\n"
                    f"</runtime-model-context>"
                )
            except Exception as e:
                logger.debug("Task runtime model context injection failed: %s", e)
            add_mission_event(mission_id, f"task#{task_id}", "task_created", f"Task started: {content[:200]}")
        except Exception as e:
            logger.debug("Mission context injection failed: %s", e)

    # ── Context Isolation: build agent-appropriate context ──────────
    agent_type = task.get("agent_type") or "analyst"

    # (A) Agent execution history: load this agent's recent completed tasks
    agent_history_ctx = ""
    if user_id and user_id != 0:
        try:
            prev_tasks = _query(
                "SELECT id, content, result, tool_log, completed_at FROM telegram_tasks "
                "WHERE user_id = %s AND agent_type = %s AND status = 'done' "
                "AND id != %s ORDER BY completed_at DESC LIMIT 3",
                (user_id, agent_type, task_id),
            )
            if prev_tasks:
                prev_tasks.reverse()  # chronological → oldest first
                num_tasks = len(prev_tasks)
                parts = []
                for idx, pt in enumerate(prev_tasks):
                    recency = num_tasks - 1 - idx  # 0=oldest, num_tasks-1=newest
                    pt_id = pt["id"]
                    pt_completed = str(pt.get("completed_at") or "?")[:19]
                    pt_summary = _extract_summary(str(pt.get("result") or ""), 500)
                    pt_tool_log = str(pt.get("tool_log") or "")

                    # Observation masking: newest=full, middle=actions only, oldest=summary only
                    if recency == 0 and num_tasks > 2:
                        # Oldest: summary only, no tool log
                        pt_tool_log = ""
                    elif recency < num_tasks - 1 and pt_tool_log:
                        # Middle: mask tool outputs (keep action lines, remove "→ ..." results)
                        masked_lines = []
                        for line in pt_tool_log.split("\n"):
                            arrow_pos = line.find(" → ")
                            if arrow_pos > 0:
                                masked_lines.append(line[:arrow_pos] + " → [masked]")
                            else:
                                masked_lines.append(line)
                        pt_tool_log = "\n".join(masked_lines)[:4000]
                    elif pt_tool_log:
                        # Newest: full tool log
                        pt_tool_log = pt_tool_log[:8000]

                    block = f"  <prev-task id=\"{pt_id}\" completed=\"{pt_completed}\">\n"
                    block += f"    <summary>{pt_summary}</summary>\n"
                    if pt_tool_log:
                        block += f"    <tool-log>\n{pt_tool_log}\n    </tool-log>\n"
                    block += f"  </prev-task>"
                    parts.append(block)
                agent_history_ctx = (
                    f"<agent-execution-history agent=\"{agent_type}\">\n"
                    + "\n".join(parts)
                    + "\n</agent-execution-history>"
                )
        except Exception as e:
            logger.debug("Agent execution history load failed: %s", e)

    # (B) Recent chat: brief high-level context (what user discussed with orchestrator)
    chat_ctx = ""
    if user_id and user_id != 0:
        try:
            recent_rows = _query(
                "SELECT role, content FROM ("
                "  SELECT role, content, id FROM telegram_chat_history"
                "  WHERE user_id = %s ORDER BY id DESC LIMIT 10"
                ") sub ORDER BY id ASC",
                (user_id,),
            )
            if recent_rows:
                chat_lines = []
                for r in recent_rows:
                    text = str(r["content"] or "")
                    if text.startswith("[SYSTEM]"):
                        # Convert system markers to neutral info (not instructions)
                        # e.g. "[SYSTEM] 사용자가 /restart ..." → "[시스템] 서비스 재시작됨 (...)"
                        import re as _re
                        ts_match = _re.search(r"\((\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} KST)\)", text)
                        ts = ts_match.group(1) if ts_match else ""
                        if "재시작 완료" in text or "재시작 시작" in text:
                            chat_lines.append(f"  [시스템] 서비스 재시작됨 ({ts}). 이미 반영 완료.")
                        elif "/deploy" in text:
                            chat_lines.append(f"  [시스템] 배포 실행됨 ({ts}). 이미 반영 완료.")
                        elif "/restart" in text:
                            chat_lines.append(f"  [시스템] 서비스 재시작됨 ({ts}). 이미 반영 완료.")
                        continue
                    role_label = "사용자" if r["role"] == "user" else "레닌"
                    chat_lines.append(f"  [{role_label}] {text[:300]}")
                if chat_lines:
                    chat_ctx = "<recent-chat>\n" + "\n".join(chat_lines) + "\n</recent-chat>"
        except Exception as e:
            logger.debug("Chat context injection for task failed: %s", e)

    # (C) Current state block (what's done, in-progress, pending)
    state_ctx = ""
    if user_id and user_id != 0:
        try:
            state_ctx = build_current_state(user_id)
        except Exception as e:
            logger.debug("Current state build failed: %s", e)

    # (D) Build full context: all parts combined
    # Note: scratchpad is no longer injected as context (replaced by current_state +
    # agent-execution-history + mission events). It's kept only for recovery marker counting.
    context_parts = []
    if state_ctx:
        context_parts.append(state_ctx)
    if mission_ctx:
        context_parts.append(mission_ctx)
    if agent_history_ctx:
        context_parts.append(agent_history_ctx)
    if chat_ctx:
        context_parts.append(chat_ctx)

    if context_parts:
        content = "\n\n".join(context_parts) + f"\n\n<task>\n{content}\n</task>"

    max_retries = 10
    retry_delay = 60

    for attempt in range(max_retries):
        try:
            bt = {}
            report = await chat_with_tools_fn(
                [{"role": "user", "content": content}],
                system_prompt=task_system_prompt,
                model=await get_model_fn(),
                max_tokens=max_tokens_task,
                budget_usd=budget_usd,
                extra_tools=extra_tools,
                extra_handlers=extra_handlers,
                on_progress=on_progress,
                budget_tracker=bt,
            )

            # Save tool execution log for agent context isolation
            tool_details = bt.get("tool_work_details", [])
            if tool_details:
                tool_log_text = "\n".join(str(d)[:500] for d in tool_details)[:20000]
                try:
                    await asyncio.to_thread(
                        _execute,
                        "UPDATE telegram_tasks SET tool_log = %s WHERE id = %s",
                        (tool_log_text, task_id),
                    )
                except Exception as e:
                    logger.debug("Failed to save tool_log for task %d: %s", task_id, e)

            # Record task completion to mission (generous summary for context chain)
            if mission_id:
                try:
                    from telegram_mission import add_mission_event
                    agent_label = f" [{task.get('agent_type', 'analyst')}]" if task.get("agent_type") else ""
                    summary = _extract_summary(report, 1500)
                    add_mission_event(
                        mission_id, f"task#{task_id}", "task_completed",
                        f"Done{agent_label}: {summary}",
                    )
                except Exception:
                    pass

            # Auto-save scout reports to Knowledge Graph
            if task.get("agent_type") == "scout":
                try:
                    from shared import process_scout_report_to_kg
                    kg_result = await asyncio.to_thread(
                        process_scout_report_to_kg,
                        report=report,
                        task_content=content,
                        agent_type="scout",
                    )
                    if kg_result.get("status") == "ok":
                        logger.info(
                            "[SCOUT→KG] Task #%d saved to KG | group=%s | facts=%d",
                            task_id,
                            kg_result.get("group_id", "?"),
                            kg_result.get("facts_count", 0),
                        )
                    else:
                        logger.debug(
                            "[SCOUT→KG] Task #%d KG save skipped: %s",
                            task_id, kg_result.get("message", "unknown reason")
                        )
                except Exception as e:
                    # Non-fatal: log but don't fail the task
                    logger.warning("[SCOUT→KG] Task #%d KG processing failed: %s", task_id, e)

            # Save full report to DB
            await asyncio.to_thread(
                _execute,
                "UPDATE telegram_tasks SET status = 'done', result = %s, "
                "completed_at = NOW() WHERE id = %s",
                (report, task_id),
            )

            # Classify priority
            priority = _classify_priority(content, report)
            priority_icon = {"high": "🔴", "normal": "🟡", "low": "🟢"}.get(priority, "🟡")

            # Send report as Markdown file
            filename = f"report_task_{task_id}.md"
            doc = BufferedInputFile(report.encode("utf-8"), filename=filename)
            summary = _extract_summary(report)
            origin = " (자율 생성)" if is_self_generated else ""
            caption = f"{priority_icon} 태스크 [{task_id}]{origin} 완료\n\n{summary}"

            if is_self_generated:
                for uid in allowed_user_ids:
                    try:
                        await bot.send_document(chat_id=uid, document=doc, caption=caption)
                    except Exception:
                        pass
            else:
                await bot.send_document(chat_id=user_id, document=doc, caption=caption)

            # Visualizer: auto-send generated images as photos
            if task.get("agent_type") == "visualizer":
                try:
                    import re as _re
                    # Extract local_path from tool log or report
                    tool_log_text = str(bt.get("tool_work_details", ""))
                    paths = _re.findall(r"local_path:\s*(/\S+\.png)", tool_log_text + "\n" + report)
                    for img_path in paths[:5]:  # max 5 images
                        if os.path.isfile(img_path):
                            with open(img_path, "rb") as f:
                                photo = BufferedInputFile(f.read(), filename=os.path.basename(img_path))
                            target = user_id if not is_self_generated else next(iter(allowed_user_ids), 0)
                            if target:
                                await bot.send_photo(chat_id=target, photo=photo, caption=f"🎨 [{task_id}] 생성 이미지")
                except Exception as e:
                    logger.debug("Visualizer auto-send image failed: %s", e)

            # Notify orchestrator of completion
            if on_complete:
                try:
                    agent_label = f" [{task.get('agent_type', 'analyst')}]" if task.get("agent_type") else ""
                    on_complete(task_id, "done", f"{agent_label} {summary}")
                except Exception:
                    logger.debug("on_complete callback failed for task %d", task_id)

            return  # success

        except Exception as e:
            err_str = str(e).lower()
            is_rate_limit = (
                "rate_limit" in err_str or
                "overloaded" in err_str or
                "529" in err_str or
                "429" in err_str or
                "too many requests" in err_str
            )

            if is_rate_limit and attempt < max_retries - 1:
                logger.warning("Task %d rate limited (attempt %d/%d), retrying in %ds...", task_id, attempt + 1, max_retries, retry_delay)
                await asyncio.sleep(retry_delay)
                continue

            # Final failure or non-rate-limit error
            logger.error("Task %d failed: %s", task_id, e)
            # Record failure to mission
            if mission_id:
                try:
                    from telegram_mission import add_mission_event
                    add_mission_event(mission_id, f"task#{task_id}", "task_completed", f"Failed: {str(e)[:500]}")
                except Exception:
                    pass
            await asyncio.to_thread(
                log_event_fn, "error", "task",
                f"Task {task_id} failed: {e}",
                detail=content[:500], task_id=task_id,
            )
            await asyncio.to_thread(
                _execute,
                "UPDATE telegram_tasks SET status = 'failed', result = %s, "
                "completed_at = NOW() WHERE id = %s",
                (str(e), task_id),
            )
            error_msg = f"❌ 태스크 [{task_id}] 실패:\n{e}"
            if is_self_generated:
                await broadcast(bot, error_msg, allowed_user_ids)
            else:
                await bot.send_message(chat_id=user_id, text=error_msg)

            # Notify orchestrator of failure
            if on_complete:
                try:
                    on_complete(task_id, "failed", str(e)[:300])
                except Exception:
                    logger.debug("on_complete callback failed for task %d", task_id)


async def recover_processing_tasks_on_startup(
    stale_minutes: int = 60,
    max_resume_attempts: int = _DEFAULT_MAX_RESUME_ATTEMPTS,
) -> dict:
    """Recover interrupted tasks at startup.

    - Recent processing tasks are NOT resumed in-place.
      Instead, they are handed off to a new child task (pending).
    - Old processing tasks are auto-closed as failed to avoid surprise re-execution.
    - Tasks repeatedly interrupted across restarts are auto-closed as failed.
    """
    try:
        stale_minutes = max(5, min(24 * 60, int(stale_minutes)))
        max_resume_attempts = max(1, min(10, int(max_resume_attempts)))

        processing_rows = await asyncio.to_thread(
            _query,
            "SELECT id, user_id, content, depth, created_at, scratchpad, mission_id, agent_type FROM telegram_tasks "
            "WHERE status = 'processing' AND completed_at IS NULL "
            "ORDER BY created_at ASC",
        )
        if not processing_rows:
            logger.info("Startup recovery: no interrupted processing tasks")
            return {
                "resumed": 0,
                "handed_off": 0,
                "closed_stale": 0,
                "closed_repeated": 0,
                "window_minutes": stale_minutes,
                "max_resume_attempts": max_resume_attempts,
            }

        handed_off = 0
        closed_stale = 0
        closed_repeated = 0

        from shared import KST
        now_kst = datetime.now(KST)

        for row in processing_rows:
            task_id = row["id"]
            user_id = int(row.get("user_id") or 0)
            content = str(row.get("content") or "")
            depth = int(row.get("depth") or 0)
            created_at = row.get("created_at")
            scratchpad = str(row.get("scratchpad") or "")
            handoff_count = scratchpad.count(_STARTUP_HANDOFF_MARKER)

            age_minutes = 0.0
            if created_at is not None:
                try:
                    # Ensure both datetimes are timezone-aware for comparison
                    ca = created_at if created_at.tzinfo else created_at.replace(tzinfo=timezone.utc)
                    age_minutes = max(0.0, (now_kst - ca).total_seconds() / 60.0)
                except Exception:
                    age_minutes = float(stale_minutes + 1)

            if age_minutes >= stale_minutes:
                await asyncio.to_thread(
                    _execute,
                    "UPDATE telegram_tasks SET status = 'failed', "
                    "result = COALESCE(result, '') || %s, completed_at = NOW() "
                    "WHERE id = %s",
                    ("\n[AUTO-CLOSED] stale processing task after restart; not resumed automatically.", task_id),
                )
                closed_stale += 1
                continue

            if handoff_count >= max_resume_attempts or depth >= _MAX_TASK_CHAIN_DEPTH:
                await asyncio.to_thread(
                    _execute,
                    "UPDATE telegram_tasks SET status = 'failed', "
                    "result = COALESCE(result, '') || %s, completed_at = NOW() "
                    "WHERE id = %s",
                    (
                        f"\n[AUTO-CLOSED] processing task repeatedly interrupted across restarts "
                        f"(handoff_count={handoff_count}, limit={max_resume_attempts}, depth={depth}).",
                        task_id,
                    ),
                )
                closed_repeated += 1
                continue

            ts = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
            handoff_note = (
                f"{_STARTUP_HANDOFF_MARKER}\n"
                f"- from_task_id: {task_id}\n"
                f"- at: {ts}\n"
                f"- reason: service restarted while task was processing\n"
                f"- handoff_attempt: {handoff_count + 1}/{max_resume_attempts}"
            )
            child_scratchpad = f"{scratchpad}\n\n{handoff_note}".strip() if scratchpad else handoff_note
            if len(child_scratchpad) > _SCRATCHPAD_MAX_CHARS:
                child_scratchpad = child_scratchpad[-_SCRATCHPAD_MAX_CHARS:]

            task_mission_id = row.get("mission_id")
            task_agent_type = row.get("agent_type")
            child_content = content
            if _RESTART_COMPLETED_MARKER not in child_content:
                child_content = f"{_RESTART_COMPLETED_MARKER}\n{child_content}"

            child_rows = await asyncio.to_thread(
                _query,
                "INSERT INTO telegram_tasks (user_id, content, status, parent_task_id, scratchpad, depth, mission_id, agent_type) "
                "VALUES (%s, %s, 'pending', %s, %s, %s, %s, %s) RETURNING id",
                (user_id, child_content, task_id, child_scratchpad, depth + 1, task_mission_id, task_agent_type),
            )
            child_id = child_rows[0]["id"] if child_rows else None

            # Record handoff to mission timeline
            if task_mission_id:
                try:
                    from telegram_mission import add_mission_event
                    add_mission_event(
                        task_mission_id, "system", "decision",
                        f"Service restart already completed: task #{task_id} → child #{child_id} (handoff {handoff_count+1}/{max_resume_attempts}); child must only perform post-restart verification"
                    )
                except Exception:
                    pass

            await asyncio.to_thread(
                _execute,
                "UPDATE telegram_tasks SET status = 'handed_off', "
                "result = COALESCE(result, '') || %s, completed_at = NOW() "
                "WHERE id = %s",
                (
                    f"\n[AUTO-HANDOFF] interrupted by restart; continued in child task #{child_id}.",
                    task_id,
                ),
            )
            handed_off += 1

        if closed_stale or closed_repeated or handed_off:
            logger.warning(
                "Startup recovery: handed_off=%d, closed_stale=%d, closed_repeated=%d "
                "(window=%dmin, resume_limit=%d)",
                handed_off, closed_stale, closed_repeated, stale_minutes, max_resume_attempts,
            )
        else:
            logger.info("Startup recovery: no interrupted processing tasks")
        return {
            "resumed": handed_off,  # backward-compatible key for existing callers
            "handed_off": handed_off,
            "closed_stale": closed_stale,
            "closed_repeated": closed_repeated,
            "window_minutes": stale_minutes,
            "max_resume_attempts": max_resume_attempts,
        }
    except Exception as e:
        logger.error("Failed to recover processing tasks on startup: %s", e)
        return {
            "resumed": 0,
            "handed_off": 0,
            "closed_stale": 0,
            "closed_repeated": 0,
            "window_minutes": stale_minutes,
            "max_resume_attempts": max_resume_attempts,
            "error": str(e),
        }


async def checkpoint_task_on_shutdown(task_id: int) -> bool:
    """Persist a last-moment checkpoint for an in-flight task before shutdown."""
    try:
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        note = (
            f"{_SHUTDOWN_CHECKPOINT_MARKER}\n"
            f"- task_id: {task_id}\n"
            f"- at: {ts}\n"
            "- note: service received SIGTERM while task was processing"
        )
        # Write to scratchpad (for startup recovery marker counting)
        await asyncio.to_thread(_append_task_scratchpad, task_id, note)
        # Also record to mission timeline (use task's own mission_id)
        try:
            task_rows = _query("SELECT mission_id FROM telegram_tasks WHERE id = %s", (task_id,))
            task_mid = task_rows[0].get("mission_id") if task_rows else None
            if task_mid:
                from telegram_mission import add_mission_event
                add_mission_event(
                    task_mid, f"task#{task_id}", "decision",
                    f"Service shutdown — task #{task_id} interrupted at {ts}"
                )
        except Exception:
            pass  # best-effort
        return True
    except Exception as e:
        logger.error("Failed to checkpoint task %s on shutdown: %s", task_id, e)
        return False


# ── Broadcast ────────────────────────────────────────────────────────

async def broadcast(bot: Bot, text: str, allowed_user_ids: set[int]):
    """Send a message to all allowed users. For system event notifications."""
    for uid in allowed_user_ids:
        try:
            await bot.send_message(chat_id=uid, text=text)
        except Exception as e:
            logger.warning("Broadcast to %s failed: %s", uid, e)


# ── System Monitor ───────────────────────────────────────────────────

async def system_monitor(
    bot: Bot,
    *,
    allowed_user_ids: set[int],
    add_alert_fn,
    clear_alert_fn,
):
    """Background loop: monitor system events and broadcast notifications."""
    from shared import get_kg_service

    # 1. Initial KG check (startup notification is handled by bot_main)
    await asyncio.sleep(10)
    kg = await asyncio.to_thread(get_kg_service)
    kg_is_up = kg is not None
    if not kg_is_up:
        add_alert_fn("KG (Neo4j) 연결 불가 — 그래프 검색/쓰기 사용 불가")

    # 2. Periodic KG health check (every 2 minutes)
    kg_was_up = kg_is_up
    while True:
        await asyncio.sleep(120)
        try:
            kg = await asyncio.to_thread(get_kg_service)
            kg_is_up = kg is not None

            if kg_was_up and not kg_is_up:
                clear_alert_fn("KG 재연결")
                add_alert_fn("KG (Neo4j) 연결 끊김 — 그래프 검색/쓰기 사용 불가")
                await broadcast(bot, "🔴 *KG 연결 끊김* — Neo4j에 연결할 수 없습니다.", allowed_user_ids)
            elif not kg_was_up and kg_is_up:
                clear_alert_fn("KG")
                add_alert_fn("KG 재연결 성공 — Neo4j 정상")
                await broadcast(bot, "🟢 *KG 재연결 성공* — Neo4j 연결이 복구되었습니다.", allowed_user_ids)

            kg_was_up = kg_is_up
        except Exception as e:
            logger.error("System monitor error: %s", e)


# ── Task Worker ──────────────────────────────────────────────────────

async def task_worker(bot: Bot, *, process_task_fn, runtime_state: dict | None = None):
    """Poll DB for pending tasks and process them one at a time.

    Args:
        bot: Telegram Bot instance.
        process_task_fn: Async callable(bot, task) to process each task.
        runtime_state: Optional mutable dict for tracking in-flight task.
    """
    logger.info("Task worker started")
    while True:
        try:
            task = await asyncio.to_thread(
                _query_one,
                "UPDATE telegram_tasks SET status = 'processing' "
                "WHERE id = (SELECT id FROM telegram_tasks WHERE status = 'pending' "
                "ORDER BY created_at LIMIT 1 FOR UPDATE SKIP LOCKED) "
                "RETURNING id, user_id, content, scratchpad, parent_task_id, depth, mission_id, agent_type",
            )
            if task:
                if runtime_state is not None:
                    runtime_state["current_task_id"] = task.get("id")
                try:
                    await process_task_fn(bot, task)
                finally:
                    if runtime_state is not None:
                        runtime_state["current_task_id"] = None
            else:
                await asyncio.sleep(5)
        except Exception as e:
            if runtime_state is not None:
                runtime_state["current_task_id"] = None
            logger.error("Worker loop error: %s", e)
            await asyncio.sleep(10)


# ── Schedule Worker ──────────────────────────────────────────────────

async def schedule_worker(bot: Bot, *, allowed_user_ids: set[int]):
    """Check cron schedules every 60s, create tasks when due."""
    from croniter import croniter
    from shared import KST

    logger.info("Schedule worker started")
    await asyncio.sleep(10)
    while True:
        try:
            schedules = await asyncio.to_thread(
                _query,
                "SELECT id, user_id, content, cron_expr, last_run_at "
                "FROM telegram_schedules WHERE enabled = TRUE",
            )
            now_kst = datetime.now(KST)
            for sched in schedules:
                try:
                    cron = croniter(sched["cron_expr"], now_kst)
                    prev_fire = cron.get_prev(datetime)
                    last_run = sched["last_run_at"]
                    # First run: only fire if prev_fire is after created_at (not immediately on registration)
                    if last_run is None:
                        created = sched.get("created_at")
                        if created and prev_fire <= created:
                            continue
                    elif prev_fire <= last_run:
                        continue
                        # Detect agent_type from [agent] prefix in content
                        sched_content = sched["content"]
                        sched_agent = None
                        if sched_content.startswith("[") and "]" in sched_content[:20]:
                            tag = sched_content[1:sched_content.index("]")].strip().lower()
                            from agents import agent_names
                            if tag in agent_names():
                                sched_agent = tag
                        await asyncio.to_thread(
                            _execute,
                            "INSERT INTO telegram_tasks (user_id, content, agent_type) VALUES (%s, %s, %s)",
                            (sched["user_id"], sched_content, sched_agent),
                        )
                        await asyncio.to_thread(
                            _execute,
                            "UPDATE telegram_schedules SET last_run_at = %s WHERE id = %s",
                            (now_kst, sched["id"]),
                        )
                        logger.info("Schedule #%d fired → task created: %.50s", sched["id"], sched["content"])
                        try:
                            await bot.send_message(
                                chat_id=sched["user_id"],
                                text=f"⏰ 스케줄 [{sched['id']}] 실행 → 태스크 생성됨\n{sched['content'][:100]}",
                            )
                        except Exception:
                            pass
                except Exception as e:
                    logger.error("Schedule #%d check error: %s", sched["id"], e)
        except Exception as e:
            logger.error("Schedule worker error: %s", e)
        await asyncio.sleep(60)


# ── Deploy Detection ─────────────────────────────────────────────────

_DEPLOY_META_PATH = "/tmp/leninbot-deploy-meta.json"


async def check_deploy_meta(bot: Bot, *, add_alert_fn):
    """On startup, check if we were just deployed. Inject into system alerts."""
    try:
        if not os.path.isfile(_DEPLOY_META_PATH):
            return
        with open(_DEPLOY_META_PATH, "r") as f:
            meta = json.load(f)
        os.remove(_DEPLOY_META_PATH)

        status = meta.get("status", "success")

        if status == "failed":
            error = meta.get("error", "unknown")
            exit_code = meta.get("exit_code", "?")
            alert_msg = f"Deploy 실패 (exit {exit_code}): {error}"
            add_alert_fn(alert_msg)
            logger.error("Deploy FAILED: exit=%s error=%s", exit_code, error)
            return

        changes = meta.get("changes", "")
        new_commit = meta.get("new_commit", "")[:7]
        prev_commit = meta.get("prev_commit", "")[:7]
        deps = " (의존성 업데이트됨)" if meta.get("deps_updated") else ""

        alert_msg = (
            f"Deploy 완료: {prev_commit}→{new_commit}{deps}. "
            f"변경: {changes}"
        )
        add_alert_fn(alert_msg)
        logger.info("Deploy detected: %s → %s", prev_commit, new_commit)
    except Exception as e:
        logger.warning("Deploy meta check failed: %s", e)
