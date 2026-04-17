"""browser_worker.py — Separate process for Browser Agent execution.

Listens on a Unix Domain Socket (/tmp/leninbot-browser.sock) for task requests
from the main Telegram bot process. Executes browser tasks using the BROWSER
AgentSpec and returns results via the socket.

Run as: python browser_worker.py
Systemd: leninbot-browser.service (MemoryMax=2G)
"""

import os
import sys
import json
import asyncio
import logging
import signal
from datetime import datetime
from pathlib import Path
import importlib

BROWSER_PROVIDER = "claude"
BROWSER_MODEL = os.getenv("BROWSER_MODEL", "claude-sonnet-4-6")


def _normalize_browser_model(raw_model: str | None) -> str:
    model = str(raw_model or "").strip()
    if not model:
        return "claude-sonnet-4-6"

    lowered = model.lower()
    if lowered in {"high", "medium", "low"}:
        tier_map = {
            "high": "claude-opus-4-7",
            "medium": "claude-sonnet-4-6",
            "low": "claude-haiku-4-5",
        }
        return tier_map[lowered]

    if lowered in {"opus", "sonnet", "haiku"}:
        alias_map = {
            "opus": "claude-opus-4-7",
            "sonnet": "claude-sonnet-4-6",
            "haiku": "claude-haiku-4-5",
        }
        return alias_map[lowered]

    if lowered.startswith("gpt") or lowered.startswith("o"):
        # logger not yet initialized at module level; use print for early warning
        print(
            f"[browser_worker] WARNING: non-Claude model override '{model}'; forcing {BROWSER_MODEL}"
        )
        return BROWSER_MODEL

    return model

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

SOCKET_PATH = "/tmp/leninbot-browser.sock"

# ── Lazy singletons (avoid import-time side effects) ─────────────────

_claude_client = None
_tools = None
_tool_handlers = None


def _init_claude_client():
    global _claude_client
    if _claude_client is not None:
        return _claude_client
    import anthropic
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    _claude_client = anthropic.AsyncAnthropic(api_key=api_key)
    return _claude_client


def _init_tools(force_reload: bool = False):
    """Load tool definitions and handlers once.

    force_reload=True is used per task so the long-lived browser worker does not
    keep serving stale tool registries after a telegram-only restart/deploy.
    """
    global _tools, _tool_handlers
    if _tools is not None and not force_reload:
        return _tools, _tool_handlers
    import telegram_tools as telegram_tools_module
    if force_reload:
        telegram_tools_module = importlib.reload(telegram_tools_module)
    _tools = telegram_tools_module.TOOLS
    _tool_handlers = telegram_tools_module.TOOL_HANDLERS
    return _tools, _tool_handlers


# ── Task Execution ───────────────────────────────────────────────────

async def execute_browser_task(task: dict) -> dict:
    """Execute a browser task and return the result.

    Mirrors the logic of _process_task_wrapper + process_task in telegram_bot.py,
    but without Telegram bot dependencies.

    Returns: {parent_id, status, result_summary, error?}
    """
    # Import task runtime dependencies lazily inside the worker process.
    # Earlier task #329 only inspected source code and reconstructed tool lists.
    # That missed the real failure mode: the long-lived browser worker can keep
    # serving with stale pre-patch modules even after telegram service restart,
    # because it is a separate process with its own import cache and lifecycle.
    from agents import get_agent
    from agents.base import AgentSpec
    from claude_loop import chat_with_tools, dedupe_tools_by_name
    from self_tools import build_task_context_tools
    from telegram_tools import MISSION_TOOL, build_mission_handler
    from telegram_tasks import process_task, build_current_state
    from shared import KST
    from db import query as db_query, execute as db_execute, query_one as db_query_one

    task_id = task.get("id")
    user_id = task.get("user_id")
    if task_id is None or user_id is None:
        return {"status": "error", "error": "Missing required fields: id, user_id"}
    agent_type = task.get("agent_type") or "browser"

    # Set per-coroutine context so tools (upload_to_r2 etc.) can identify the task
    try:
        from telegram_bot import current_task_ctx
        current_task_ctx.set({"task_id": task_id, "agent_type": agent_type})
    except Exception:
        pass

    logger.info(
        "Processing browser task #%d for user %d (provider=%s, model=%s)",
        task_id,
        user_id,
        BROWSER_PROVIDER,
        BROWSER_MODEL,
    )

    try:
        spec = get_agent(agent_type)
    except (ValueError, ImportError):
        spec = get_agent("browser")

    # Load and filter tools. Force module reload here because browser_worker is a
    # separate long-lived process; a telegram service restart does not refresh its
    # already-imported tool registry.
    all_tools, all_handlers = _init_tools(force_reload=True)
    agent_tools, agent_handlers = spec.filter_tools(all_tools, all_handlers)

    # Add task-context tools
    ctx_tools, ctx_handlers = build_task_context_tools(
        task_id, user_id, task.get("depth", 0),
        mission_id=task.get("mission_id"),
    )
    agent_tools.extend(ctx_tools)
    agent_handlers.update(ctx_handlers)

    # Bind mission handler without re-adding schema.
    # MISSION_TOOL is already present in the base tool registry, and re-appending it
    # creates duplicate tool names in the API payload (task #326 failure).
    if "mission" in {t.get("name") for t in agent_tools}:
        agent_handlers["mission"] = build_mission_handler(user_id)

    # Final safety net: task-context / future registry composition must never send
    # duplicate tool names to the upstream API.
    agent_tools = dedupe_tools_by_name(agent_tools)

    # Render system prompt
    system_prompt = spec.render_prompt(
        current_datetime=datetime.now(KST).strftime("%Y-%m-%d %H:%M KST"),
        system_alerts="",
    )

    client = _init_claude_client()

    # Build the chat_with_tools_fn closure matching _chat_with_tools signature
    async def _chat_fn(
        messages,
        max_rounds=None,
        system_prompt=None,
        model=None,
        max_tokens=None,
        budget_usd=None,
        extra_tools=None,
        extra_handlers=None,
        on_progress=None,
        budget_tracker=None,
        task_id=None,
        finalization_tools=None,
    ):
        # Use only agent-filtered tools/handlers — not the full set
        merged_tools = list(extra_tools or [])
        merged_handlers = dict(extra_handlers or {})
        # Browser worker always runs on Claude. Runtime /config provider can be
        # switched to OpenAI for chat/task orchestration, but this worker uses the
        # Anthropic tool-calling loop and Anthropic client only. Guard here against
        # any caller accidentally passing OpenAI model IDs or generic tiers.
        resolved_model = _normalize_browser_model(model or BROWSER_MODEL)

        return await chat_with_tools(
            messages,
            client=client,
            model=resolved_model,
            tools=merged_tools,
            tool_handlers=merged_handlers,
            system_prompt=system_prompt or "",
            max_rounds=max_rounds or spec.max_rounds,
            max_tokens=max_tokens or 8192,
            log_event=_log_event,
            budget_usd=budget_usd or spec.budget_usd,
            on_progress=on_progress,
            budget_tracker=budget_tracker,
            task_id=task_id,
            finalization_tools=finalization_tools,
        )

    async def _get_model():
        # Keep browser worker provider/model pinned to Claude regardless of global
        # runtime provider toggles. process_task() uses this for progress metadata
        # and any fallback model lookup.
        return _normalize_browser_model(BROWSER_MODEL)

    # Create a dummy bot-like object that skips Telegram sends
    class _NullBot:
        """Stub that absorbs Telegram API calls without actually sending."""
        async def send_message(self, *a, **kw):
            pass
        async def send_document(self, *a, **kw):
            pass
        async def send_photo(self, *a, **kw):
            pass

    # Get allowed_user_ids from env
    allowed_user_ids = {
        int(uid.strip())
        for uid in os.getenv("ALLOWED_USER_IDS", "").split(",")
        if uid.strip()
    }

    try:
        await process_task(
            _NullBot(),
            task,
            chat_with_tools_fn=_chat_fn,
            get_model_fn=_get_model,
            task_system_prompt=system_prompt,
            max_tokens_task=4096,
            allowed_user_ids=allowed_user_ids,
            log_event_fn=_log_event,
            extra_tools=agent_tools,
            extra_handlers=agent_handlers,
            budget_usd=spec.budget_usd,
        )

        # Read final result from DB
        row = db_query_one(
            "SELECT status, result FROM telegram_tasks WHERE id = %s",
            (task_id,),
        )
        status = (row or {}).get("status", "done")
        result = (row or {}).get("result", "")
        summary = (result[:500] if result else "completed")

        logger.info("Browser task #%d finished: %s", task_id, status)
        resp = {
            "parent_id": task.get("parent_task_id"),
            "status": status,
            "result_summary": summary,
        }
        if status == "failed":
            resp["error"] = result[:1000] if result else "task failed (no detail)"
        return resp

    except Exception as e:
        logger.error("Browser task #%d failed: %s", task_id, e, exc_info=True)
        # Mark task as failed in DB
        try:
            db_execute(
                "UPDATE telegram_tasks SET status = 'failed', result = %s, completed_at = NOW() "
                "WHERE id = %s AND status IN ('processing', 'queued')",
                (f"browser_worker error: {e}", task_id),
            )
        except Exception:
            pass
        return {
            "parent_id": task.get("parent_task_id"),
            "status": "failed",
            "result_summary": str(e)[:500],
            "error": str(e),
        }


def _log_event(level, source, message, detail=None, task_id=None):
    """Persist error/warning to DB (same as telegram_bot._log_event)."""
    try:
        from db import execute as db_execute
        db_execute(
            "INSERT INTO telegram_error_log (level, source, message, detail, task_id) "
            "VALUES (%s, %s, %s, %s, %s)",
            (level[:10], source[:100], message[:2000], detail[:4000] if detail else None, task_id),
        )
    except Exception as e:
        logger.warning("_log_event DB write failed: %s", e)


# ── Unix Domain Socket Server ────────────────────────────────────────

async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    """Handle a single connection from the main process."""
    peer = writer.get_extra_info("peername") or "unix"
    try:
        # Read until EOF — the main process sends JSON then closes write end
        raw = await asyncio.wait_for(reader.read(1024 * 1024), timeout=10)
        if not raw:
            writer.close()
            await writer.wait_closed()
            return

        msg = json.loads(raw.decode("utf-8"))
        cmd = msg.get("cmd", "task")

        if cmd == "ping":
            response = {"status": "alive"}
        elif cmd == "task":
            response = await execute_browser_task(msg)
        else:
            response = {"status": "error", "error": f"unknown cmd: {cmd}"}

    except asyncio.TimeoutError:
        response = {"status": "error", "error": "read timeout"}
    except json.JSONDecodeError as e:
        response = {"status": "error", "error": f"invalid JSON: {e}"}
    except Exception as e:
        logger.error("handle_client error: %s", e, exc_info=True)
        response = {"status": "error", "error": str(e)}

    try:
        writer.write(json.dumps(response, ensure_ascii=False).encode("utf-8"))
        await writer.drain()
        writer.close()
        await writer.wait_closed()
    except Exception:
        pass


async def main():
    # Clean up stale socket (race-safe)
    try:
        os.unlink(SOCKET_PATH)
    except FileNotFoundError:
        pass

    server = await asyncio.start_unix_server(handle_client, path=SOCKET_PATH)
    os.chmod(SOCKET_PATH, 0o660)

    logger.info("Browser worker listening on %s", SOCKET_PATH)

    # Graceful shutdown on SIGTERM
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _on_signal():
        logger.info("Received shutdown signal")
        stop_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _on_signal)

    async with server:
        await stop_event.wait()

    logger.info("Browser worker shutting down")
    try:
        os.unlink(SOCKET_PATH)
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    asyncio.run(main())
