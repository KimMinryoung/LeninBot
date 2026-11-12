"""Fire-and-forget tasks that are neither garbage-collected nor silent.

The event loop keeps only a weak reference to a task, so a bare
``asyncio.create_task(...)`` whose result is dropped can be collected before it
finishes, and an exception in it surfaces only as "Task exception was never
retrieved" at collection time. ``spawn`` holds the task until it is done and
logs a failure with the task's name.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine

logger = logging.getLogger(__name__)

_tasks: set[asyncio.Task] = set()


def _done(task: asyncio.Task) -> None:
    _tasks.discard(task)
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("background task %s failed", task.get_name(), exc_info=exc)


def spawn(coro: Coroutine, *, name: str | None = None) -> asyncio.Task:
    task = asyncio.get_running_loop().create_task(coro, name=name)
    _tasks.add(task)
    task.add_done_callback(_done)
    return task


def track(task: asyncio.Task) -> asyncio.Task:
    """Hold and log a task created elsewhere (e.g. from a signal handler's loop)."""
    _tasks.add(task)
    task.add_done_callback(_done)
    return task
