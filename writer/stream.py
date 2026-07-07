"""Writer generation runs and their SSE streams.

A run is a detached background task registered in writer.runs; the HTTP stream
is only an observer. Clients reattach via stream_active_run after drops or
page reloads, and the final event always persists even with no subscriber."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Awaitable, Callable

from claude_loop import chat_with_tools
from tool_gateway.security import CallerContext, caller_scope

from writer.config import (
    WRITER_CRITIC_MAX_ROUNDS,
    WRITER_CRITIC_MAX_TOKENS,
    WRITER_DEFAULT_MAX_TOKENS,
    WRITER_IDLE_TIMEOUT_SEC,
    WRITER_MAX_ROUNDS,
    WRITER_PROVIDER_IDLE_TIMEOUT_SEC,
)
from writer.models import WRITER_CRITIC_CHOICE, resolve_light_model, resolve_writer_model
from writer.prompts import (
    build_critic_system_blocks,
    build_system_blocks,
    critic_user_message,
    messages_for_model,
    parse_writer_response,
    writer_error_message,
)
from writer.runs import WriterRun, get_active_run, register_run, unregister_run
from writer.store import get_manuscript, get_project, insert_message, touch_project
from writer.tools import build_critic_tools, build_writer_tools

logger = logging.getLogger(__name__)

_writer_background_tasks: set[asyncio.Task] = set()


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _consume_run(
    run: WriterRun,
    queue: asyncio.Queue[str | None],
    client_disconnected: Callable[[], Awaitable[bool]] | None,
) -> AsyncIterator[str]:
    """Relay a run's broadcast queue to one SSE consumer, with keepalive pings
    and idle cutoff. The background run outlives any consumer."""
    subscribed_at = asyncio.get_running_loop().time()
    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=15)
            except asyncio.TimeoutError:
                if client_disconnected is not None and await client_disconnected():
                    logger.warning(
                        "writer stream subscriber disconnected project_id=%s reason=request.is_disconnected",
                        run.project_id,
                    )
                    return
                # Idle is measured against this subscriber's own attach time as
                # well, so every reconnect gets a full fresh grace window.
                idle_for = asyncio.get_running_loop().time() - max(run.last_progress, subscribed_at)
                if idle_for >= WRITER_IDLE_TIMEOUT_SEC:
                    logger.warning(
                        "writer stream idle project_id=%s idle_for=%ss; background run remains active",
                        run.project_id,
                        int(idle_for),
                    )
                    # Not an error: the run is alive but silent. The client
                    # treats this as a cue to reattach, not as a final event.
                    yield _sse({
                        "type": "stream_idle",
                        "content": (
                            f"No model progress for {int(idle_for)}s. The server-side writer run "
                            "is still active; reconnecting to it."
                        ),
                    })
                    return
                yield _sse({"type": "ping"})
                continue
            if item is None:
                return
            yield item
    except asyncio.CancelledError:
        logger.warning(
            "writer stream subscriber disconnected project_id=%s reason=streaming_response_cancelled",
            run.project_id,
        )
        return
    finally:
        run.subscribers.discard(queue)


async def stream_active_run(
    *,
    project_id: int,
    client_disconnected: Callable[[], Awaitable[bool]] | None = None,
) -> AsyncIterator[str]:
    """Reattach to a live background writer run (e.g. after a page reload or a
    dropped stream). Replays the text streamed so far, then follows live."""
    run = get_active_run(project_id)
    if run is None:
        yield _sse({"type": "no_active_run"})
        return
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    run.subscribers.add(queue)
    yield _sse({
        "type": "run_status",
        "run_id": run.run_id,
        "user_message_id": run.user_message_id,
        "model": run.model,
        "model_display": run.model_display,
        "text_snapshot": run.text_snapshot(),
    })
    if run.final_event is not None:
        run.subscribers.discard(queue)
        yield _sse(run.final_event)
        return
    async for item in _consume_run(run, queue, client_disconnected):
        yield item


async def stream_writer_reply(
    *,
    project_id: int,
    prompt: str,
    selection_start: int | None = None,
    selection_end: int | None = None,
    model_choice: str | None = None,
    critic: bool = False,
    client_disconnected: Callable[[], Awaitable[bool]] | None = None,
) -> AsyncIterator[str]:
    project = get_project(project_id)
    if not project:
        yield _sse({"type": "error", "content": "Project not found."})
        return

    try:
        writer_client, writer_model, writer_display, model_extra = resolve_writer_model(model_choice)
    except (ValueError, RuntimeError) as exc:
        yield _sse({"type": "error", "content": str(exc)})
        return

    request_kind = ""
    model_messages = messages_for_model(project_id, prompt, selection_start, selection_end)
    user_row = insert_message(
        project_id=project_id,
        role="user",
        content=prompt.strip(),
        request_kind=request_kind,
    )

    run = WriterRun(
        project_id=project_id,
        run_id=uuid.uuid4().hex,
        user_message_id=user_row.get("id"),
        model=writer_model,
        model_display=writer_display,
    )
    subscriber_queue: asyncio.Queue[str | None] = asyncio.Queue()
    run.subscribers.add(subscriber_queue)
    answer_holder: list[str] = []
    error_holder: list[str] = []
    budget_tracker: dict = {}
    critic_holder: list[str] = []
    critic_budget_tracker: dict = {}
    critic_edit_count = 0
    # The critic (퇴고) is easy work: it runs on the light DeepSeek tier, not
    # the heavy main model, falling back to the main model when unconfigured.
    critic_client, critic_model, critic_display, critic_extra = resolve_light_model(
        WRITER_CRITIC_CHOICE, (writer_client, writer_model, writer_display, model_extra)
    )
    persisted = False

    writer_tools, writer_handlers = build_writer_tools(project_id)

    def persist_result() -> dict:
        nonlocal persisted
        if persisted:
            return {"type": "error", "content": "Writer result was already persisted."}
        persisted = True
        if error_holder:
            error_text = writer_error_message(writer_display, error_holder[0])
            assistant_row = insert_message(
                project_id=project_id,
                role="assistant",
                content=f"<commentary>\n{error_text}\n</commentary>",
                request_kind=request_kind,
                model=writer_model,
                stop_reason="error",
                usage=budget_tracker.get("usage") or {},
                cost_usd=budget_tracker.get("total_cost"),
            )
            touch_project(project_id)
            return {
                "type": "error",
                "message_id": assistant_row.get("id"),
                "content": error_text,
                "edits": [dict(edit) for edit in run.edits],
            }

        final_text = (answer_holder[0] if answer_holder else "").strip()
        critic_ran = bool(critic_holder)
        if critic_ran:
            main_parsed = parse_writer_response(final_text)
            critic_parsed = parse_writer_response(critic_holder[0].strip())
            commentary = main_parsed["commentary_text"] or main_parsed["manuscript_text"]
            critic_note = critic_parsed["commentary_text"] or critic_parsed["manuscript_text"]
            if critic_note:
                commentary = (commentary + "\n\n" if commentary else "") + "[퇴고] " + critic_note
            final_text = "<commentary>\n" + commentary + "\n</commentary>"
        parsed_response = parse_writer_response(final_text)
        usage = dict(budget_tracker.get("usage") or {})
        cost = budget_tracker.get("total_cost")
        stop_reason = str(budget_tracker.get("stop_reason") or "")
        if critic_ran:
            for key, value in (critic_budget_tracker.get("usage") or {}).items():
                if isinstance(value, (int, float)) and isinstance(usage.get(key, 0), (int, float)):
                    usage[key] = usage.get(key, 0) + value
                elif key not in usage:
                    usage[key] = value
            critic_cost = critic_budget_tracker.get("total_cost")
            if critic_cost is not None:
                cost = (cost or 0.0) + critic_cost
        # Delegated light-agent sub-runs (research_web) recorded on the run.
        for extra_cost in run.extra_costs:
            cost = (cost or 0.0) + extra_cost["cost_usd"]
        assistant_row = insert_message(
            project_id=project_id,
            role="assistant",
            content=final_text,
            request_kind=request_kind,
            model=writer_model,
            stop_reason=stop_reason,
            usage=usage,
            cost_usd=cost,
        )
        touch_project(project_id)
        event = {
            "type": "done",
            "message_id": assistant_row.get("id"),
            "model": writer_model,
            "model_display": writer_display,
            "stop_reason": stop_reason,
            "usage": usage,
            "cost_usd": cost,
            "manuscript_text": parsed_response["manuscript_text"],
            "commentary_text": parsed_response["commentary_text"],
            "display_text": parsed_response["display_text"],
            "edits": [dict(edit) for edit in run.edits],
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
        if critic_ran:
            event["critic"] = {
                "ran": True,
                "model": critic_model,
                "model_display": critic_display,
                "cost_usd": critic_budget_tracker.get("total_cost"),
                "edit_count": critic_edit_count,
            }
        return event

    async def on_progress(event: str, detail: str):
        run.last_progress = asyncio.get_running_loop().time()
        if event == "text_delta" and detail:
            run.append_text(detail)
            await run.broadcast(_sse({"type": "text_delta", "content": detail}))
        elif event == "budget" and detail:
            await run.broadcast(_sse({"type": "budget", "content": detail}))
        elif event in ("tool_call", "tool_result") and detail:
            await run.broadcast(_sse({"type": "tool", "content": detail}))
        elif event == "provider_retry" and detail:
            await run.broadcast(_sse({"type": "provider_retry", "content": detail}))

    async def run_critic_pass() -> None:
        """Optional second-pass line editor (퇴고) over this turn's changed
        spans. Failures are isolated: the main result always persists."""
        nonlocal critic_edit_count
        edits_snapshot = [dict(edit) for edit in run.edits]
        body = str((get_manuscript(project_id) or {}).get("body") or "")
        critic_msg = critic_user_message(body, edits_snapshot)
        if not critic_msg:
            return
        await run.broadcast(_sse({"type": "critic_start"}))
        await run.broadcast(_sse({"type": "tool", "content": f"Line-edit pass (퇴고, {critic_display})…"}))
        edits_before = len(run.edits)
        try:
            critic_tools, critic_handlers = build_critic_tools(project_id)
            critic_system = await asyncio.to_thread(build_critic_system_blocks, project)
            with caller_scope(CallerContext(interface="system", agent_name="writer", is_owner=True)):
                critic_text = await chat_with_tools(
                    [{"role": "user", "content": critic_msg}],
                    client=critic_client,
                    model=critic_model,
                    tools=critic_tools,
                    tool_handlers=critic_handlers,
                    system_prompt=critic_system,
                    max_rounds=WRITER_CRITIC_MAX_ROUNDS,
                    max_tokens=WRITER_CRITIC_MAX_TOKENS,
                    budget_usd=100.0,
                    budget_tracker=critic_budget_tracker,
                    on_progress=on_progress,
                    agent_name="writer_critic",
                    provider_idle_timeout_sec=WRITER_PROVIDER_IDLE_TIMEOUT_SEC,
                    **critic_extra,
                )
            critic_holder.append(critic_text)
        except Exception:
            logger.exception("writer critic pass failed project_id=%s", project_id)
            critic_holder.append(
                "<commentary>\n퇴고 패스가 실패했습니다 — 본문 초안과 그 편집 내용은 저장되어 있습니다.\n</commentary>"
            )
        finally:
            critic_edit_count = len(run.edits) - edits_before

    async def run_llm() -> None:
        try:
            system_blocks = await asyncio.to_thread(
                build_system_blocks, project, project_id, selection_start, selection_end
            )
            with caller_scope(CallerContext(interface="system", agent_name="writer", is_owner=True)):
                result = await chat_with_tools(
                    model_messages,
                    client=writer_client,
                    model=writer_model,
                    tools=writer_tools,
                    tool_handlers=writer_handlers,
                    system_prompt=system_blocks,
                    max_rounds=WRITER_MAX_ROUNDS,
                    max_tokens=WRITER_DEFAULT_MAX_TOKENS,
                    budget_usd=100.0,
                    budget_tracker=budget_tracker,
                    on_progress=on_progress,
                    agent_name="writer",
                    provider_idle_timeout_sec=WRITER_PROVIDER_IDLE_TIMEOUT_SEC,
                    **model_extra,
                )
            answer_holder.append(result)
            if critic:
                await run_critic_pass()
        except asyncio.CancelledError:
            # Server shutdown/restart while the run was in flight. Persist an
            # explanatory message in the finally block, then re-raise.
            error_holder.append("writer run cancelled by server shutdown")
            raise
        except Exception as exc:
            logger.exception("writer model request failed project_id=%s", project_id)
            error_holder.append(str(exc))
        finally:
            try:
                event = persist_result()
                run.final_event = event
                await run.broadcast(_sse(event))
                logger.info(
                    "writer finalized background run project_id=%s event=%s message_id=%s connected_subscribers=%s",
                    project_id,
                    event.get("type"),
                    event.get("message_id"),
                    len(run.subscribers),
                )
            except Exception:
                logger.exception("writer failed to persist background run project_id=%s", project_id)
                await run.broadcast(_sse({
                    "type": "error",
                    "content": "Writer finished but failed to save the result. Check server logs.",
                }))
            finally:
                await run.broadcast(None)
                unregister_run(run)

    # Register and start the run BEFORE the first yield: a client that drops
    # right after connecting must not prevent the background run from starting.
    register_run(run)
    task = asyncio.create_task(run_llm())
    _writer_background_tasks.add(task)
    task.add_done_callback(_writer_background_tasks.discard)
    yield _sse({"type": "user_saved", "message_id": user_row.get("id"), "run_id": run.run_id})
    async for item in _consume_run(run, subscriber_queue, client_disconnected):
        yield item
