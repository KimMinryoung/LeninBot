"""leninbot-worker: executes queued agent tasks (worker/endpoint.py queues them).

Polls agent_worker_tasks, runs up to WORKER_CONCURRENCY tasks at once, and
stops a running task whose row was cancelled. A task interrupted by a restart
is picked up again once its lease expires (store.MAX_ATTEMPTS).
"""
from __future__ import annotations

import asyncio
import logging
import os

from worker import runner, store

logger = logging.getLogger("worker")

CONCURRENCY = int(os.getenv("WORKER_CONCURRENCY", "2"))
POLL_SECONDS = 2.0


async def execute_commulingo(task: dict) -> None:
    from worker import commulingo
    output = await commulingo.run(task["id"], task["request"])
    error = output.pop("error", None)
    await asyncio.to_thread(store.finish, task["id"], status="failed" if error else "done", result=output,
                            usage=output["usage"], rejections=output["metrics"].get("rejections", [])[-20:],
                            error=error[:2000] if error else None)


async def execute(task: dict) -> None:
    task_id = task["id"]
    try:
        if task["request"].get("kind", "generic") != "generic":
            await execute_commulingo(task)
            return
        outcome = await runner.run(task_id, task["request"])
        await asyncio.to_thread(store.finish, task_id, status="done", result={
            "value": outcome["result"], "validation": outcome["validation"]},
            sources=outcome["sources"], usage=outcome["usage"], rejections=outcome["rejections"])
    except asyncio.CancelledError:
        logger.info("worker task %s cancelled", task_id)
        raise
    except runner.TaskFailed as exc:
        report = exc.report
        await asyncio.to_thread(store.finish, task_id, status="failed", sources=report.get("sources", ()),
                                usage=report.get("usage"), rejections=report.get("rejections", ()), error=str(exc)[:2000])
    except Exception as exc:  # noqa: BLE001 - one task's crash must not stop the worker
        logger.exception("worker task %s crashed", task_id)
        await asyncio.to_thread(store.finish, task_id, status="failed", error=f"worker error: {exc}"[:2000])


async def poll_once(running: dict) -> None:
    for task_id, job in list(running.items()):
        if await asyncio.to_thread(store.is_cancelled, task_id):
            job.cancel()
    while len(running) < CONCURRENCY:
        task = await asyncio.to_thread(store.claim)
        if task is None:
            return
        logger.info("worker task %s started (client %s, attempt %s)", task["id"], task["client"], task["attempts"])
        job = asyncio.create_task(execute(task))
        running[task["id"]] = job
        job.add_done_callback(lambda _job, task_id=task["id"]: running.pop(task_id, None))


def expire_sources() -> int:
    """Drop expired page bodies from the CommuLingo session source cache."""
    from commulingo.pipeline.store import Store
    return Store().expire_sources()


async def main() -> None:
    running: dict[int, asyncio.Task] = {}
    logger.info("leninbot-worker started (concurrency %s)", CONCURRENCY)
    last_expiry = 0.0
    while True:
        try:
            await poll_once(running)
            now = asyncio.get_running_loop().time()
            if now - last_expiry > 3600:
                last_expiry = now
                await asyncio.to_thread(expire_sources)
        except Exception:  # noqa: BLE001 - a DB hiccup is retried on the next poll
            logger.exception("worker poll failed")
        await asyncio.sleep(POLL_SECONDS)


if __name__ == "__main__":
    from services.api_common import setup_service_logging
    setup_service_logging(quiet_neo4j=True)
    asyncio.run(main())
