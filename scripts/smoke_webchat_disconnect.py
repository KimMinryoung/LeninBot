#!/usr/bin/env python3
"""Verify that a detached web-chat SSE observer does not cancel its run."""

from __future__ import annotations

import asyncio
import json
from contextlib import ExitStack
from unittest.mock import AsyncMock, patch
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


async def _check_detached_run_persists(*, regenerate: bool = False, detach: bool = True, save_fails: bool = False) -> None:
    import services.web_chat as web_chat

    originals = {
        name: getattr(web_chat, name)
        for name in (
            "get_persona",
            "_build_persona_tools",
            "_load_web_history",
            "resolve_runtime_profile",
            "_load_web_feedback_rows",
            "_load_web_tone_policy",
            "_render_web_feedback_context",
            "_render_web_tone_policy",
            "render_system_prompt",
            "chat_with_tools",
            "_log_chat",
            "_deepseek_anthropic_client",
        )
    }
    patches = ExitStack()
    patches.enter_context(patch("services.web_chat._reserve_chat_log_id", return_value=4242))
    patches.enter_context(patch("kg_runtime.recall.entity_gated_kg_block", return_value=""))
    patches.enter_context(patch("llm.provider_failover.resolve_deepseek_failover_model", new=AsyncMock(return_value=None)))
    history_loader = patches.enter_context(patch("services.web_chat._load_web_history", return_value=[]))
    lookup = patches.enter_context(patch("services.web_chat.get_web_chat_log_for_feedback", return_value={
        "id": 4242, "session_id": "smoke-detached-session", "fingerprint": "original-fingerprint",
        "user_query": "original question", "bot_answer": "old answer",
    }))
    saved_feedback = patches.enter_context(patch("services.web_chat.save_web_chat_feedback"))
    updated = patches.enter_context(patch("services.web_chat._update_chat_answer", return_value=None if save_fails else 4242))
    original_redis = sys.modules.get("redis_state")
    original_to_thread = asyncio.to_thread
    release_model = asyncio.Event()
    model_started = asyncio.Event()
    persisted: list[dict] = []
    log_feedback_ids: list[list[int]] = []
    cancelled: list[bool] = []

    async def fake_profile(*args, **kwargs):
        return SimpleNamespace(
            provider="deepseek",
            model_id="smoke-model",
            tier="medium",
            budget_usd=0.3,
            max_rounds=2,
            max_tokens=128,
            display_name="Smoke Model",
        )

    async def fake_chat(*args, **kwargs):
        model_started.set()
        try:
            await release_model.wait()
        except asyncio.CancelledError:
            cancelled.append(True)
            raise
        return "detached answer"

    def fake_log(*args, **kwargs):
        persisted.append({"session_id": kwargs["session_id"], "answer": kwargs["bot_answer"]})
        log_feedback_ids.append(kwargs.get("feedback_ids", []))
        return None if save_fails else 4242

    async def immediate_to_thread(function, /, *args, **kwargs):
        return function(*args, **kwargs)

    try:
        asyncio.to_thread = immediate_to_thread
        web_chat.get_persona = lambda persona: SimpleNamespace(
            id="cyber-lenin", provider_override=None, tier_override=None
        )
        web_chat._build_persona_tools = lambda spec: ([], {})
        web_chat.resolve_runtime_profile = fake_profile
        web_chat._load_web_feedback_rows = lambda *args, **kwargs: [{"id": 11, "note": "once"}]
        web_chat._load_web_tone_policy = lambda *args, **kwargs: []
        web_chat._render_web_feedback_context = lambda *args, **kwargs: ""
        web_chat._render_web_tone_policy = lambda *args, **kwargs: ""
        web_chat.render_system_prompt = lambda *args, **kwargs: "system"
        web_chat.chat_with_tools = fake_chat
        web_chat._log_chat = fake_log
        web_chat._deepseek_anthropic_client = object()
        sys.modules["redis_state"] = SimpleNamespace(
            register_active_web_chat=lambda *args, **kwargs: None,
            unregister_active_web_chat=lambda *args, **kwargs: None,
        )

        stream = web_chat.handle_web_chat(
            message="disconnect smoke",
            session_id="smoke-detached-session",
            fingerprint="smoke-fingerprint",
            user_agent="smoke",
            ip_address="127.0.0.1",
            regenerate_from_id=4242 if regenerate else None,
            feedback_note="rewrite once" if regenerate else "",
        )
        first = await anext(stream)
        assert '"type": "run_started"' in first
        await asyncio.wait_for(model_started.wait(), timeout=2)
        if detach:
            await asyncio.wait_for(stream.aclose(), timeout=2)
            assert web_chat.has_active_web_chat_run("smoke-detached-session")
            assert web_chat.detached_web_chat_run_count() == 1
        release_model.set()
        if detach:
            await asyncio.wait_for(asyncio.gather(*web_chat._web_chat_background_tasks), timeout=2)
        else:
            events = [json.loads(event.removeprefix("data: ")) async for event in stream]
            terminal = [event for event in events if event["type"] in {"answer", "error"}]
            assert len(terminal) == 1
            answer = terminal[0]
            assert answer["type"] == ("error" if save_fails else "answer")
            if not save_fails:
                assert answer["message_id"] == 4242
                assert answer["regenerated_from_id"] == (4242 if regenerate else None)
            assert answer["request_id"] == json.loads(first.removeprefix("data: "))["request_id"]

        assert cancelled == []
        if regenerate:
            assert persisted == []
            lookup.assert_called_once_with(4242, ["smoke-fingerprint"], "smoke-detached-session", "cyber-lenin", account_user_id=None)
            updated.assert_called_once()
            assert updated.call_args.kwargs["chat_log_id"] == 4242
            assert updated.call_args.kwargs["fingerprint"] == "original-fingerprint"
            assert updated.call_args.kwargs["bot_answer"] == "detached answer"
            assert history_loader.call_args.args[4] == {4242}
            assert saved_feedback.call_args.kwargs["pending"] is False
            assert log_feedback_ids == []
        else:
            assert persisted == [{"session_id": "smoke-detached-session", "answer": "detached answer"}]
            updated.assert_not_called()
            assert log_feedback_ids == [[11]]
        assert not web_chat.has_active_web_chat_run("smoke-detached-session")
    finally:
        release_model.set()
        asyncio.to_thread = original_to_thread
        if web_chat._web_chat_background_tasks:
            await asyncio.gather(*web_chat._web_chat_background_tasks, return_exceptions=True)
        patches.close()
        for name, value in originals.items():
            setattr(web_chat, name, value)
        if original_redis is None:
            sys.modules.pop("redis_state", None)
        else:
            sys.modules["redis_state"] = original_redis


async def _check_vector_timeout() -> None:
    import services.web_chat as web_chat

    original_handler = web_chat.TOOL_HANDLERS["vector_search"]
    original_timeout = web_chat._WEBCHAT_VECTOR_SEARCH_TIMEOUT_SEC

    async def slow_vector_search(**kwargs):
        await asyncio.sleep(1)
        return "late result"

    try:
        web_chat.TOOL_HANDLERS["vector_search"] = slow_vector_search
        web_chat._WEBCHAT_VECTOR_SEARCH_TIMEOUT_SEC = 0.01
        _, handlers = web_chat._build_persona_tools({"vector_search"})
        result = await handlers["vector_search"](query="timeout smoke")
        assert result.startswith("Vector search timed out")
    finally:
        web_chat.TOOL_HANDLERS["vector_search"] = original_handler
        web_chat._WEBCHAT_VECTOR_SEARCH_TIMEOUT_SEC = original_timeout


@patch("db.query", new=lambda *args, **kwargs: [])
@patch("db.query_one", new=lambda *args, **kwargs: None)
@patch("db.execute", new=lambda *args, **kwargs: None)
def main() -> int:
    asyncio.run(_check_detached_run_persists())
    asyncio.run(_check_detached_run_persists(regenerate=True))
    asyncio.run(_check_detached_run_persists(regenerate=True, detach=False))
    asyncio.run(_check_detached_run_persists(detach=False, save_fails=True))
    asyncio.run(_check_detached_run_persists(regenerate=True, detach=False, save_fails=True))
    asyncio.run(_check_detached_run_persists(save_fails=True))
    asyncio.run(_check_vector_timeout())
    print("webchat disconnect smoke ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
