#!/usr/bin/env python3
"""Refuse service restarts while user-visible work is in progress.

This is deliberately a guard, not a resume system. It prevents accidental
loss of in-flight Telegram tasks or web chat answer generation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")


def _db_active_tasks() -> list[dict]:
    try:
        from db import query

        return query(
            "SELECT id, status, agent_type, left(content, 120) AS content "
            "FROM telegram_tasks "
            "WHERE completed_at IS NULL AND status IN ('processing', 'queued') "
            "ORDER BY id DESC LIMIT 20"
        )
    except Exception as e:
        return [{"warning": f"db task check failed: {e}"}]


def _redis_active_tasks() -> list[dict]:
    try:
        from redis_state import get_all_active_tasks

        return get_all_active_tasks()
    except Exception as e:
        return [{"warning": f"redis task check failed: {e}"}]


def _redis_active_web_chats() -> list[dict]:
    try:
        from redis_state import get_active_web_chats

        return get_active_web_chats()
    except Exception as e:
        return [{"warning": f"redis web chat check failed: {e}"}]


def _format_task(task: dict) -> str:
    if task.get("error"):
        return f"- {task['error']}"
    if task.get("warning"):
        return f"- warning: {task['warning']}"
    tid = task.get("task_id") or task.get("id") or "?"
    status = task.get("status", "?")
    agent = task.get("agent_type") or task.get("agent") or "?"
    content = (task.get("content") or "").replace("\n", " ")
    return f"- task #{tid} status={status} agent={agent} {content[:120]}".rstrip()


def _format_web_chat(chat: dict) -> str:
    if chat.get("error"):
        return f"- {chat['error']}"
    if chat.get("warning"):
        return f"- warning: {chat['warning']}"
    return (
        f"- web_chat request={chat.get('request_id', '?')} "
        f"session={chat.get('session_id', '?')} started_at={chat.get('started_at', '?')}"
    )


def check_restart_allowed(service: str) -> tuple[bool, list[str]]:
    blockers: list[str] = []
    warnings: list[str] = []
    if service in {"telegram", "all"}:
        redis_tasks = _redis_active_tasks()
        db_tasks = _db_active_tasks()
        active_tasks = [t for t in (redis_tasks + db_tasks) if not t.get("warning")]
        warnings.extend(_format_task(t) for t in (redis_tasks + db_tasks) if t.get("warning"))
        if active_tasks:
            blockers.append("Active Telegram/background tasks:")
            blockers.extend(_format_task(t) for t in active_tasks)

    if service in {"api", "all"}:
        web_chats = _redis_active_web_chats()
        warnings.extend(_format_web_chat(c) for c in web_chats if c.get("warning"))
        web_chats = [c for c in web_chats if not c.get("warning")]
        if web_chats:
            blockers.append("Active web chat answer generation:")
            blockers.extend(_format_web_chat(c) for c in web_chats)

    if warnings:
        blockers.append("Guard warnings:")
        blockers.extend(warnings)
        if len(blockers) == len(warnings) + 1:
            return True, blockers

    return not blockers, blockers


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("service", choices=["telegram", "api", "all"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    allowed, blockers = check_restart_allowed(args.service)
    if allowed or args.force:
        if blockers and args.force:
            print("Restart guard overridden with --force. Blockers were:")
            print("\n".join(blockers))
        return 0

    print(f"Restart refused for {args.service}: active work is in progress.")
    print("\n".join(blockers))
    print("Use --force only if losing that in-flight work is acceptable.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
