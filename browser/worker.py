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

BROWSER_MODEL_OVERRIDE = os.getenv("BROWSER_MODEL", "").strip() or None
BROWSER_PROVIDER_OVERRIDE = os.getenv("BROWSER_PROVIDER", "").strip().lower() or None


def _normalize_browser_model(raw_model: str | None, provider: str = "deepseek") -> str:
    model = str(raw_model or "").strip()
    if not model:
        return "deepseek-v4-flash" if provider == "deepseek" else "gpt-5.5-mini"

    lowered = model.lower()
    if lowered in {"high", "medium", "low"}:
        if provider == "deepseek":
            tier_map = {"high": "deepseek-v4-pro", "medium": "deepseek-v4-flash", "low": "deepseek-v4-flash"}
        elif provider == "openai":
            tier_map = {"high": "gpt-5.5", "medium": "gpt-5.5-mini", "low": "gpt-5.5-nano"}
        else:
            tier_map = {"high": "deepseek-v4-pro", "medium": "deepseek-v4-flash", "low": "deepseek-v4-flash"}
        return tier_map[lowered]

    if lowered in {"opus", "sonnet", "haiku"} or lowered.startswith("claude"):
        print(f"[browser_worker] WARNING: Claude model override '{model}' ignored for browser worker")
        return "deepseek-v4-flash"

    return model


def _resolve_browser_provider(raw_provider: str | None) -> str:
    provider = str(raw_provider or "").strip().lower()
    if provider in {"deepseek", "openai"}:
        return provider
    if provider == "claude":
        logger.warning("Browser worker forbids Claude provider; using deepseek instead")
    return "deepseek"

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

SOCKET_PATH = "/tmp/leninbot-browser.sock"

# ── Lazy singletons (avoid import-time side effects) ─────────────────

_provider_clients: dict[str, object] = {}
_tools = None
_tool_handlers = None


def _init_provider_client(provider: str):
    if provider in _provider_clients:
        return _provider_clients[provider]
    from secrets_loader import get_secret

    if provider == "openai":
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=get_secret("OPENAI_API_KEY", "") or "")
    else:
        import anthropic
        client = anthropic.AsyncAnthropic(
            api_key=get_secret("DEEPSEEK_API_KEY", "") or "",
            base_url=os.getenv(
                "DEEPSEEK_ANTHROPIC_BASE_URL",
                "https://api.deepseek.com/anthropic",
            ).rstrip("/"),
        )
    _provider_clients[provider] = client
    return client


def _init_tools(force_reload: bool = False):
    """Load tool definitions and handlers once.

    force_reload=True is used per task so the long-lived browser worker does not
    keep serving stale tool registries after a telegram-only restart/deploy.
    """
    global _tools, _tool_handlers
    if _tools is not None and not force_reload:
        return _tools, _tool_handlers
    import runtime_tools.registry as runtime_tools_registry
    if force_reload:
        runtime_tools_registry = importlib.reload(runtime_tools_registry)
    _tools = runtime_tools_registry.TOOLS
    _tool_handlers = runtime_tools_registry.TOOL_HANDLERS
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
    from claude_loop import dedupe_tools_by_name
    from self_runtime.tools import build_task_context_tools
    from runtime_tools.registry import MISSION_TOOL, build_mission_handler
    from telegram.tasks import process_task, build_current_state
    from db import query as db_query, execute as db_execute, query_one as db_query_one

    task_id = task.get("id")
    user_id = task.get("user_id")
    if task_id is None or user_id is None:
        return {"status": "error", "error": "Missing required fields: id, user_id"}
    agent_type = task.get("agent_type") or "browser"

    # Set per-coroutine context so tools (upload_to_r2 etc.) can identify the task
    try:
        from telegram.bot import current_task_ctx
        current_task_ctx.set({"task_id": task_id, "agent_type": agent_type})
    except Exception:
        pass

    try:
        spec = get_agent(agent_type)
    except (ValueError, ImportError):
        spec = get_agent("browser")

    provider = _resolve_browser_provider(BROWSER_PROVIDER_OVERRIDE or spec.effective_provider("deepseek"))
    browser_model = _normalize_browser_model(BROWSER_MODEL_OVERRIDE or spec.model, provider)

    logger.info(
        "Processing browser task #%d for user %d (provider=%s, model=%s)",
        task_id,
        user_id,
        provider,
        browser_model,
    )

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

    # Render system prompt in the selected non-Claude provider format. Prompt is
    # fully static post-refactor; timestamp is injected into the user message.
    system_prompt = spec.render_prompt(provider=provider)

    client = _init_provider_client(provider)

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
        terminal_tools=None,
    ):
        # Use only agent-filtered tools/handlers — not the full set
        merged_tools = list(extra_tools or [])
        merged_handlers = dict(extra_handlers or {})
        resolved_model = _normalize_browser_model(model or browser_model, provider)

        # Inject the runtime header (current time + model) into the task's user
        # message so the system prompt stays byte-stable across invocations and
        # prompt caching hits. System alerts are a Telegram-chat concept and
        # intentionally skipped for this background worker.
        from telegram.bot import (
            _build_runtime_prelude, _join_context_blocks,
            _merge_runtime_context_into_last_user,
        )
        messages = _merge_runtime_context_into_last_user(
            messages,
            _join_context_blocks(_build_runtime_prelude(provider, kind="task")),
        )

        if provider == "deepseek":
            from claude_loop import chat_with_tools as deepseek_chat
            return await deepseek_chat(
                messages,
                client=client,
                model=resolved_model,
                tools=merged_tools,
                tool_handlers=merged_handlers,
                system_prompt=system_prompt or "",
                max_rounds=max_rounds or spec.max_rounds,
                max_tokens=max_tokens or 8192,
                budget_usd=budget_usd or spec.budget_usd,
                on_progress=on_progress,
                budget_tracker=budget_tracker,
                task_id=task_id,
                finalization_tools=finalization_tools,
                terminal_tools=terminal_tools,
                thinking={"type": "disabled"},
            )

        from openai_tool_loop import chat_with_tools as openai_chat
        return await openai_chat(
            messages,
            client=client,
            model=resolved_model,
            tools=merged_tools,
            tool_handlers=merged_handlers,
            system_prompt=system_prompt or "",
            max_rounds=max_rounds or spec.max_rounds,
            max_tokens=max_tokens or 8192,
            budget_usd=budget_usd or spec.budget_usd,
            on_progress=on_progress,
            budget_tracker=budget_tracker,
            task_id=task_id,
            finalization_tools=finalization_tools,
            terminal_tools=terminal_tools,
            provider_label=provider,
        )

    async def _get_model():
        # process_task() uses this for progress metadata and fallback model lookup.
        return browser_model

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
