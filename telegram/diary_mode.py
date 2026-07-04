"""Shared diary task mode helpers."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

DEFAULT_DIARY_WRITING_PROMPT = "[diary] Write a periodic diary entry"


def normalize_diary_prompt(text: str | None) -> str:
    return " ".join(str(text or "").split())


def scheduled_diary_prompts() -> set[str]:
    prompts = {normalize_diary_prompt(DEFAULT_DIARY_WRITING_PROMPT)}
    try:
        from db import query

        rows = query(
            """
            SELECT content, agent_type
              FROM telegram_schedules
             WHERE enabled = TRUE
            """
        )
    except Exception as exc:
        logger.debug("diary schedule prompt lookup failed: %s", exc)
        return prompts

    for row in rows:
        content = str(row.get("content") or "").strip()
        agent_type = str(row.get("agent_type") or "").strip().lower()
        if agent_type == "diary" or content.startswith("[diary]"):
            normalized = normalize_diary_prompt(content)
            if normalized:
                prompts.add(normalized)
    return prompts


def is_diary_writing_task(task: dict | None, content: str | None = None) -> bool:
    task = task or {}
    if task.get("agent_type") != "diary":
        return False
    task_content = normalize_diary_prompt(content if content is not None else task.get("content"))
    return task_content in scheduled_diary_prompts()
