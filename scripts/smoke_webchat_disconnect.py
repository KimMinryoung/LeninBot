#!/usr/bin/env python3
"""Verify that a detached web-chat SSE observer does not cancel its run."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


async def _check_detached_run_persists() -> None:
    import web_chat

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
    original_redis = sys.modules.get("redis_state")
    original_to_thread = asyncio.to_thread
    release_model = asyncio.Event()
    model_started = asyncio.Event()
    persisted: list[dict] = []
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
        persisted.append({"session_id": args[0], "answer": args[5]})
        return 4242

    async def immediate_to_thread(function, /, *args, **kwargs):
        return function(*args, **kwargs)

    try:
        asyncio.to_thread = immediate_to_thread
        web_chat.get_persona = lambda persona: SimpleNamespace(
            id="cyber-lenin", provider_override=None, tier_override=None
        )
        web_chat._build_persona_tools = lambda spec: ([], {})
        web_chat._load_web_history = lambda *args, **kwargs: []
        web_chat.resolve_runtime_profile = fake_profile
        web_chat._load_web_feedback_rows = lambda *args, **kwargs: []
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
        )
        first = await anext(stream)
        assert '"type": "run_started"' in first
        await asyncio.wait_for(model_started.wait(), timeout=2)
        await asyncio.wait_for(stream.aclose(), timeout=2)

        assert web_chat.has_active_web_chat_run("smoke-detached-session")
        assert web_chat.detached_web_chat_run_count() == 1
        release_model.set()
        for _ in range(100):
            if persisted:
                break
            await asyncio.sleep(0.01)

        assert cancelled == []
        assert persisted == [{
            "session_id": "smoke-detached-session",
            "answer": "detached answer",
        }]
        assert not web_chat.has_active_web_chat_run("smoke-detached-session")
    finally:
        release_model.set()
        asyncio.to_thread = original_to_thread
        for name, value in originals.items():
            setattr(web_chat, name, value)
        if original_redis is None:
            sys.modules.pop("redis_state", None)
        else:
            sys.modules["redis_state"] = original_redis


async def _check_vector_timeout() -> None:
    import web_chat

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


def main() -> int:
    asyncio.run(_check_detached_run_persists())
    asyncio.run(_check_vector_timeout())
    print("webchat disconnect smoke ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
