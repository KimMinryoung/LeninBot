"""telegram_tasks.py — Background task processing, scheduling, and system monitoring.

Extracted from telegram_bot.py for modularity.
"""

import os
import json
import asyncio
import logging
from datetime import datetime

from aiogram import Bot
from aiogram.types import BufferedInputFile
from db import query as _query, execute as _execute, query_one as _query_one

logger = logging.getLogger(__name__)


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
):
    """Process a task: run tools, generate report, save to DB, send as file.

    Args:
        bot: Telegram Bot instance.
        task: Dict with id, user_id, content, scratchpad, parent_task_id, depth.
        chat_with_tools_fn: Async callable matching _chat_with_tools signature.
        get_model_fn: Callable returning current model ID.
        task_system_prompt: System prompt for task execution.
        max_tokens_task: Max tokens for task output.
        allowed_user_ids: Set of allowed Telegram user IDs.
        log_event_fn: Callable for persistent error logging.
        extra_tools: Additional tool definitions (e.g. task-context tools).
        extra_handlers: Additional tool handlers.
        budget_usd: USD budget for this task (default $1.00).
    """
    task_id = task["id"]
    user_id = task["user_id"]
    content = task["content"]
    scratchpad = task.get("scratchpad") or ""
    depth = task.get("depth") or 0
    parent_task_id = task.get("parent_task_id")
    is_self_generated = (user_id == 0)

    # Inject inherited context from parent scratchpad
    if scratchpad:
        content = f"## Inherited Context (from parent task #{parent_task_id}, depth={depth})\n{scratchpad}\n\n---\n\n## Task\n{content}"

    max_retries = 10
    retry_delay = 60

    for attempt in range(max_retries):
        try:
            report = await chat_with_tools_fn(
                [{"role": "user", "content": content}],
                max_rounds=50,
                system_prompt=task_system_prompt,
                model=get_model_fn(),
                max_tokens=max_tokens_task,
                budget_usd=budget_usd,
                extra_tools=extra_tools,
                extra_handlers=extra_handlers,
            )

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

    # 1. Startup notification
    await asyncio.sleep(5)
    kg = await asyncio.to_thread(get_kg_service)
    kg_status = "connected" if kg else "unavailable"
    add_alert_fn(f"Deploy 완료 — KG: {kg_status}")
    if not kg:
        add_alert_fn("KG (Neo4j AuraDB) 연결 불가 — 그래프 검색/쓰기 사용 불가")
    await broadcast(bot, (
        f"🟢 *Deploy 완료* — 새 버전이 live입니다.\n"
        f"  KG (Neo4j): {kg_status}"
    ), allowed_user_ids)

    # 2. Periodic KG health check (every 2 minutes)
    kg_was_up = kg is not None
    while True:
        await asyncio.sleep(120)
        try:
            kg = await asyncio.to_thread(get_kg_service)
            kg_is_up = kg is not None

            if kg_was_up and not kg_is_up:
                clear_alert_fn("KG 재연결")
                add_alert_fn("KG (Neo4j AuraDB) 연결 끊김 — 그래프 검색/쓰기 사용 불가")
                await broadcast(bot, "🔴 *KG 연결 끊김* — Neo4j AuraDB에 연결할 수 없습니다.", allowed_user_ids)
            elif not kg_was_up and kg_is_up:
                clear_alert_fn("KG")
                add_alert_fn("KG 재연결 성공 — Neo4j AuraDB 정상")
                await broadcast(bot, "🟢 *KG 재연결 성공* — Neo4j AuraDB 연결이 복구되었습니다.", allowed_user_ids)

            kg_was_up = kg_is_up
        except Exception as e:
            logger.error("System monitor error: %s", e)


# ── Task Worker ──────────────────────────────────────────────────────

async def task_worker(bot: Bot, *, process_task_fn):
    """Poll DB for pending tasks and process them one at a time.

    Args:
        bot: Telegram Bot instance.
        process_task_fn: Async callable(bot, task) to process each task.
    """
    logger.info("Task worker started")
    while True:
        try:
            task = await asyncio.to_thread(
                _query_one,
                "UPDATE telegram_tasks SET status = 'processing' "
                "WHERE id = (SELECT id FROM telegram_tasks WHERE status = 'pending' "
                "ORDER BY created_at LIMIT 1 FOR UPDATE SKIP LOCKED) "
                "RETURNING id, user_id, content, scratchpad, parent_task_id, depth",
            )
            if task:
                await process_task_fn(bot, task)
            else:
                await asyncio.sleep(5)
        except Exception as e:
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
                    if last_run is None or prev_fire > last_run:
                        await asyncio.to_thread(
                            _execute,
                            "INSERT INTO telegram_tasks (user_id, content) VALUES (%s, %s)",
                            (sched["user_id"], sched["content"]),
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
