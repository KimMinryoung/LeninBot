"""claude_loop.py — Claude tool-use loop (Anthropic protocol adapter).

Extracted from telegram_bot.py. Dependencies injected via function parameters
to avoid circular imports. The round/forced-final control flow is the shared
engine in agent_loop.run_tool_loop; this module contributes the Anthropic
protocol mechanics (_ClaudeProtocolAdapter).

Strict-at-source policy: inputs are normalized once via `_normalize_initial_messages`
(flattened to text-only alternating history) before the loop starts. Tool protocol
blocks only ever originate from this loop's own response parsing, which guarantees
id/name correctness, 1:1 tool_use↔tool_result pairing (safety-netted), and strict
assistant→user alternation. No post-hoc sanitization is performed.
"""

import asyncio
import json
import logging

from agent_loop import FinalTurn, Turn, run_tool_loop
from tool_loop_common import (
    build_budget_tracker, emit_progress,
    EMPTY_RESPONSE_FALLBACK,
    call_with_transient_retry,
    dedupe_tools_by_name,
    estimate_text_tokens,
    is_transient_provider_error as _is_transient_provider_error,
    provider_status_code as _provider_status_code,
)
from tool_gateway.dispatcher import execute_tool, execute_tools_batch, compact_tool_definitions
from llm.provider_registry import anthropic_pricing_table

logger = logging.getLogger(__name__)

_REPLAY_ONLY_BLOCK_TYPES = {"thinking", "redacted_thinking"}


# ── Cache TTL ────────────────────────────────────────────────────────
# Anthropic's default ephemeral cache lives 5 minutes. That's too short for a
# Telegram conversation where the user may read, think, multitask between
# turns. We use the 1-hour extended-TTL tier — the write premium goes from
# 1.25× to 2.0× base input, but any turn within an hour of the prior one
# hits cache_read (at 0.1× base) instead of paying full cache_creation again.
# Break-even after 2 reads; almost always a win for chat usage.
_CACHE_CONTROL_1H = {"type": "ephemeral", "ttl": "1h"}


# ── Pricing Constants (USD per million tokens) ──────────────────────
# Per-tier list prices. Cache-creation shown for the **1-hour TTL** tier
# (matches what this loop writes). cache_read is identical across TTL tiers.
# Prefix-match picks by base model name so pinned-date variants reuse the row.
PRICING_TABLE = anthropic_pricing_table()

# Fallback when the model string doesn't match any known family — use Sonnet
# (middle tier) so we don't wildly under- or over-report on unknown variants.
PRICING = PRICING_TABLE["claude-sonnet-5"]


def _pricing_for(model: str) -> dict:
    """Pick the pricing row for a Claude model id. Matches by prefix so
    pinned-date variants (``claude-haiku-4-5-20251001``) reuse the family."""
    pricing_table = anthropic_pricing_table()
    fallback = pricing_table["claude-sonnet-5"]
    if not model:
        return fallback
    if model in pricing_table:
        return pricing_table[model]
    for base, price in pricing_table.items():
        if model.startswith(base + "-") or model.startswith(base + "."):
            return price
    return fallback


def _calculate_cost(usage, model: str | None = None) -> float:
    """Calculate USD cost from a response.usage object for the given model."""
    p = _pricing_for(model or "")
    cost = 0.0
    cost += getattr(usage, "input_tokens", 0) * p["input"]
    cost += getattr(usage, "output_tokens", 0) * p["output"]
    # Cache tokens (may not always be present)
    cost += getattr(usage, "cache_creation_input_tokens", 0) * p["cache_creation"]
    cost += getattr(usage, "cache_read_input_tokens", 0) * p["cache_read"]
    return cost


def _usage_to_dict(usage) -> dict:
    """Return stable token usage metadata for API callers that persist costs."""
    if not usage:
        return {}
    keys = (
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
    )
    return {key: int(getattr(usage, key, 0) or 0) for key in keys}


def _update_budget_tracker(
    budget_tracker: dict | None,
    *,
    total_cost: float,
    rounds_used: int,
    was_interrupted: bool,
    tool_work_details: list,
    response=None,
    model: str | None = None,
) -> None:
    if budget_tracker is None:
        return
    metadata = build_budget_tracker(total_cost, rounds_used, was_interrupted, tool_work_details)
    if response is not None:
        metadata.update({
            "model": model or "",
            "stop_reason": str(getattr(response, "stop_reason", "") or ""),
            "usage": _usage_to_dict(getattr(response, "usage", None)),
        })
        # Refusals carry a structured stop_details (category + explanation) —
        # surface it so callers can tell the user WHY instead of a bare
        # "refusal". stop_details is None for every other stop_reason.
        details = getattr(response, "stop_details", None)
        if metadata.get("stop_reason") == "refusal" and details is not None:
            metadata["stop_details"] = {
                "category": getattr(details, "category", None),
                "explanation": getattr(details, "explanation", None),
            }
    budget_tracker.update(metadata)


# ── Content helpers ──────────────────────────────────────────────────

def _to_block_dict(block):
    """Best-effort conversion of SDK block objects to plain dict."""
    if isinstance(block, dict):
        return {k: v for k, v in block.items() if v is not None}
    if hasattr(block, "model_dump"):
        try:
            dumped = block.model_dump(exclude_none=True)
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass
    if hasattr(block, "type"):
        out = {"type": getattr(block, "type", None)}
        for key in ("id", "name", "input", "text", "tool_use_id", "content", "is_error"):
            if hasattr(block, key):
                out[key] = getattr(block, key)
        return out
    return None


def _content_block_for_replay(block: dict) -> dict | None:
    """Return the exact assistant content block shape that can be replayed.

    DeepSeek's Anthropic-compatible thinking mode requires reasoning blocks to
    be included in replayed assistant messages when a turn performs tool calls.
    Those blocks are provider protocol, not user-visible text.
    """
    if not isinstance(block, dict):
        return None
    btype = block.get("type")
    if btype in _REPLAY_ONLY_BLOCK_TYPES:
        return {k: v for k, v in block.items() if v is not None}
    if btype == "text":
        return {"type": "text", "text": str(block.get("text", ""))}
    return None


def _coerce_text(value) -> str:
    """Convert arbitrary nested content to human-readable text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [_coerce_text(v) for v in value]
        return "\n".join(p for p in parts if p).strip()
    if isinstance(value, dict):
        btype = value.get("type")
        if btype == "text":
            return str(value.get("text", ""))
        if "content" in value:
            return _coerce_text(value.get("content"))
        if "text" in value:
            return str(value.get("text", ""))
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)
    return str(value)


def _normalize_initial_messages(msgs: list[dict]) -> list[dict]:
    """Structural fix: normalize inbound history to text-only alternating chat.

    We intentionally remove all tool protocol blocks from persisted/external history.
    Tool protocol for the current request is generated only inside chat_with_tools,
    which eliminates cross-turn dangling tool_use/tool_result mismatches at the root.
    """
    clean: list[dict] = []
    for raw in msgs:
        if not isinstance(raw, dict):
            role = "user"
            text = _coerce_text(raw)
        else:
            role = raw.get("role", "user")
            if role not in ("user", "assistant"):
                role = "assistant" if role in ("model", "bot") else "user"
            text = _coerce_text(raw.get("content", ""))

        # Merge consecutive same-role into one message → strictly alternating.
        if clean and clean[-1]["role"] == role:
            clean[-1]["content"] = f"{clean[-1]['content']}\n{text}".strip()
        else:
            clean.append({"role": role, "content": text})

    return clean


def _with_message_cache_breakpoint(msgs: list[dict], max_marks: int = 2) -> list[dict]:
    """Return a shallow copy of `msgs` with cache breakpoints on stable turns.

    Prompt caching on the Messages API requires explicit breakpoints. The most
    recent assistant turn caches the current stable prefix. For long web chats,
    the recent-history window can slide; marking the first retained assistant as
    well preserves a stable early anchor instead of reducing hits to only
    system/tools.

    `max_marks` caps how many messages get marked: the Messages API allows at
    most 4 cache_control blocks per request, and system blocks + the tools
    block already consume some (the writer passes 2 cached system blocks, so
    only 1 slot remains for messages — exceeding the cap is a hard 400 on the
    Anthropic endpoint; DeepSeek's compatible endpoint doesn't enforce it,
    which is why this stayed latent until a multi-round Fable writer call).

    No-op when there is no assistant message yet (first turn) or no slots
    remain. Leaves the input list and its inner dicts untouched.
    """
    if not msgs or max_marks <= 0:
        return msgs
    result = list(msgs)

    assistant_indexes = [
        i for i, msg in enumerate(result)
        if isinstance(msg, dict) and msg.get("role") == "assistant"
    ]
    if not assistant_indexes:
        return result

    breakpoint_indexes = [assistant_indexes[-1]]
    if max_marks > 1 and len(assistant_indexes) > 1:
        breakpoint_indexes.insert(0, assistant_indexes[0])

    def _mark_message(i: int) -> None:
        msg = dict(result[i])
        content = msg.get("content", "")
        if isinstance(content, str):
            msg["content"] = [
                {"type": "text", "text": content or "(계속)",
                 "cache_control": _CACHE_CONTROL_1H}
            ]
        elif isinstance(content, list) and content:
            new_content = list(content)
            last = new_content[-1]
            last = dict(last) if isinstance(last, dict) else {"type": "text", "text": str(last)}
            last["cache_control"] = _CACHE_CONTROL_1H
            new_content[-1] = last
            msg["content"] = new_content
        else:
            msg["content"] = [
                {"type": "text", "text": "(계속)",
                 "cache_control": _CACHE_CONTROL_1H}
            ]
        result[i] = msg

    for i in breakpoint_indexes:
        _mark_message(i)
    return result


def _append_user_text_message(msgs: list[dict], text: str):
    """Append user text while preserving role alternation."""
    if msgs and msgs[-1].get("role") == "user":
        prev = msgs[-1].get("content", "")
        if isinstance(prev, str):
            msgs[-1]["content"] = f"{prev}\n{text}".strip()
        elif isinstance(prev, list):
            msgs[-1]["content"].append({"type": "text", "text": text})
        else:
            msgs[-1]["content"] = f"{_coerce_text(prev)}\n{text}".strip()
    else:
        msgs.append({"role": "user", "content": text})


# ── Token Estimation ─────────────────────────────────────────────────
# CJK-aware (tool_loop_common.estimate_text_tokens). The old flat len//3
# under-counted Korean ~3×, so the max_input_tokens ceiling and the replay
# checkpointing both fired far later than policy intended on Korean-heavy
# transcripts.
estimate_tokens = estimate_text_tokens


# ── Chat with Tools Loop ─────────────────────────────────────────────

async def _drain_stream_with_idle_guard(stream, idle_timeout_sec: float, on_progress):
    """Consume a MessageStream with an inactivity watchdog, return the final message.

    Liveness is measured on the FULL event stream, not text_stream: a round
    that answers with pure tool calls (or a long silent think) emits thinking/
    input_json deltas but zero text deltas, and the old text-only watchdog
    killed such streams as idle — observed 2026-07-07 when the writer critic
    (deepseek-v4-pro, tool-only responses) stalled out three times in a row.
    Only genuine event silence for idle_timeout_sec now trips the timeout.
    """
    event_iter = stream.__aiter__()
    while True:
        try:
            event = await asyncio.wait_for(event_iter.__anext__(), timeout=idle_timeout_sec)
        except StopAsyncIteration:
            break
        except TimeoutError as exc:
            raise TimeoutError(
                f"Provider stream produced no events for {int(idle_timeout_sec)}s"
            ) from exc
        if getattr(event, "type", "") == "text" and getattr(event, "text", ""):
            await emit_progress(on_progress, "text_delta", event.text)
    try:
        return await asyncio.wait_for(stream.get_final_message(), timeout=idle_timeout_sec)
    except TimeoutError as exc:
        raise TimeoutError(
            f"Provider stream did not finalize within "
            f"{int(idle_timeout_sec)}s after event stream ended"
        ) from exc


class _ClaudeProtocolAdapter:
    """Anthropic-protocol mechanics for agent_loop.run_tool_loop.

    Owns message shapes (content blocks, tool_use/tool_result pairing,
    replayed thinking blocks), prompt-cache breakpoints, the input-token
    ceiling with replay checkpointing, streaming with the idle guard, and
    Anthropic pricing. Control flow lives in agent_loop.run_tool_loop.
    """

    def __init__(
        self, *, client, model, tools, tool_handlers, system_prompt,
        max_tokens, max_input_tokens, recover_input_via_tools,
        on_progress, log_event, thinking, output_config,
        provider_idle_timeout_sec,
    ):
        self.client = client
        self.model = model
        self.tool_handlers = tool_handlers
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.max_input_tokens = max_input_tokens
        self.recover_input_via_tools = recover_input_via_tools
        self.on_progress = on_progress
        self.log_event = log_event
        self.thinking = thinking
        self.output_config = output_config
        self.provider_idle_timeout_sec = provider_idle_timeout_sec
        self.tool_execution_cache: dict[str, tuple[str, bool]] = {}
        self.state = None

        # Prompt caching: mark system prompt and tools as cacheable with the
        # 1-hour TTL tier (see _CACHE_CONTROL_1H rationale above). Most callers
        # use a single text prompt; writer can pass multiple cacheable system
        # blocks so stable project instructions and manuscript context can hit
        # independently.
        if isinstance(system_prompt, list):
            self.cached_system = [dict(block) for block in system_prompt]
        else:
            self.cached_system = [
                {"type": "text", "text": system_prompt, "cache_control": _CACHE_CONTROL_1H}
            ]

        # Compact verbose tool/schema descriptions before sending them to the
        # model. Names, parameter types, required keys, enums, and defaults
        # are preserved.
        cached_tools = compact_tool_definitions(tools)
        if cached_tools:
            cached_tools[-1] = {**cached_tools[-1], "cache_control": _CACHE_CONTROL_1H}
        self.cached_tools = cached_tools

        # The Messages API allows at most 4 cache_control blocks per request;
        # system blocks and the tools block already consume their share, so
        # message-level breakpoints only get whatever slots remain.
        _system_cache_blocks = sum(
            1 for b in self.cached_system if isinstance(b, dict) and b.get("cache_control")
        )
        _tools_cache_blocks = 1 if cached_tools else 0
        self.message_cache_marks = max(0, 4 - _system_cache_blocks - _tools_cache_blocks)

    def bind(self, state):
        self.state = state

    def normalize(self, messages):
        # Root-cause fix: start from text-only canonical history.
        # Tool protocol blocks are generated only within this call.
        return _normalize_initial_messages(messages)

    # ── API calls ────────────────────────────────────────────────────

    async def _call_once(self, **kwargs):
        # When on_progress is wired, route API calls through the streaming
        # interface so text deltas flow to the caller as they're generated.
        # Callers that don't care about text_delta (e.g. Telegram) simply drop
        # the event — the final Message object is identical either way.
        if self.max_input_tokens is not None:
            request_blob = json.dumps(
                {
                    "system": kwargs.get("system", []),
                    "messages": kwargs.get("messages", []),
                    "tools": kwargs.get("tools", []),
                },
                ensure_ascii=False,
                default=str,
            )
            estimated_input = estimate_tokens(request_blob)
            if estimated_input > int(self.max_input_tokens) and self.recover_input_via_tools:
                compacted = [dict(message) for message in kwargs.get("messages", [])]
                from tool_gateway.inference import is_replay_safe_tool
                tool_names_by_id = {}
                for prior in compacted:
                    prior_content = prior.get("content")
                    if not isinstance(prior_content, list):
                        continue
                    for prior_block in prior_content:
                        if isinstance(prior_block, dict) and prior_block.get("type") == "tool_use":
                            tool_names_by_id[str(prior_block.get("id") or "")] = str(
                                prior_block.get("name") or ""
                            )
                # Replay-safe read results are reproducible from their source. Keep tool names and arguments in the preceding
                # assistant tool_use block, but replace bulky result bodies
                # oldest-first with an explicit replay instruction. The latest
                # result is eligible only if older results are insufficient; order
                # naturally gives it that last-resort behavior.
                for message in compacted:
                    content = message.get("content")
                    if not isinstance(content, list):
                        continue
                    changed = False
                    blocks = []
                    for block in content:
                        if (
                            isinstance(block, dict)
                            and block.get("type") == "tool_result"
                            and is_replay_safe_tool(tool_names_by_id.get(
                                str(block.get("tool_use_id") or "")
                            ))
                            and len(str(block.get("content") or "")) > 800
                        ):
                            block = dict(block)
                            block["content"] = (
                                "[Input checkpoint: prior tool result omitted from replay. "
                                "The source remains stored; repeat the preceding read-only tool "
                                "call with the exact same tool name and arguments to recover it.]"
                            )
                            changed = True
                        blocks.append(block)
                    if changed:
                        message["content"] = blocks
                        request_blob = json.dumps(
                            {"system": kwargs.get("system", []), "messages": compacted, "tools": kwargs.get("tools", [])},
                            ensure_ascii=False, default=str,
                        )
                        estimated_input = estimate_tokens(request_blob)
                        if estimated_input <= int(self.max_input_tokens):
                            break
                kwargs["messages"] = compacted
            if estimated_input > int(self.max_input_tokens):
                raise ValueError(
                    f"Estimated input {estimated_input} tokens exceeds policy limit "
                    f"{int(self.max_input_tokens)}; compact to durable summary and anchor-based reads first."
                )
        if self.thinking is not None:
            kwargs["thinking"] = self.thinking
        if self.output_config is not None:
            kwargs["output_config"] = self.output_config
        if self.on_progress is None:
            call = self.client.messages.create(**kwargs)
            if self.provider_idle_timeout_sec:
                return await asyncio.wait_for(call, timeout=self.provider_idle_timeout_sec)
            return await call
        async with self.client.messages.stream(**kwargs) as stream:
            if self.provider_idle_timeout_sec:
                return await _drain_stream_with_idle_guard(
                    stream, self.provider_idle_timeout_sec, self.on_progress
                )
            async for text in stream.text_stream:
                if text:
                    await emit_progress(self.on_progress, "text_delta", text)
            return await stream.get_final_message()

    async def _call(self, **kwargs):
        return await call_with_transient_retry(
            lambda: self._call_once(**kwargs),
            label=self.model,
            on_progress=self.on_progress,
        )

    async def call_model(self, msgs, round_num):
        create_kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": self.cached_system,
            "messages": _with_message_cache_breakpoint(msgs, self.message_cache_marks),
        }
        if self.cached_tools:
            create_kwargs["tools"] = self.cached_tools
        return await self._call(**create_kwargs)

    # ── Cost tracking ────────────────────────────────────────────────

    def track_cost(self, response, label):
        # Log cache-token breakdown at INFO so prompt-caching effectiveness
        # is visible in journald without a debug rebuild — if cache_read
        # stays at 0 across rounds, something in the prefix is drifting and
        # the ephemeral cache can't latch on.
        usage = getattr(response, "usage", None)
        if not usage:
            return False
        cost = _calculate_cost(usage, self.model)
        self.state.add_cost(cost)
        logger.info(
            "%s usage: in=%d out=%d cache_create=%d cache_read=%d → $%.4f (total: $%.4f / $%.2f)",
            label,
            getattr(usage, "input_tokens", 0),
            getattr(usage, "output_tokens", 0),
            getattr(usage, "cache_creation_input_tokens", 0),
            getattr(usage, "cache_read_input_tokens", 0),
            cost, self.state.total_cost, self.state.budget_usd,
        )
        return True

    # ── Round parsing / message building ─────────────────────────────

    def parse_turn(self, response, round_num):
        stop_reason = getattr(response, "stop_reason", None)
        if stop_reason not in ("tool_use", "pause_turn"):
            return Turn(
                is_tool_round=False,
                text_parts=[b.text for b in response.content if b.type == "text"],
                truncated_by_length=(stop_reason == "max_tokens"),
                finish_reason=str(stop_reason or "stop"),
                raw=response,
            )

        assistant_content = []
        tool_calls: list[tuple[str, str, dict]] = []
        text_parts: list[str] = []
        for block in response.content:
            b = _to_block_dict(block) or {"type": getattr(block, "type", "unknown")}
            btype = b.get("type")

            if btype == "text":
                replay_block = _content_block_for_replay(b)
                if replay_block:
                    assistant_content.append(replay_block)
                text_parts.append(str(b.get("text", "")))
            elif btype in _REPLAY_ONLY_BLOCK_TYPES:
                replay_block = _content_block_for_replay(b)
                if replay_block:
                    assistant_content.append(replay_block)
            elif btype in ("server_tool_use", "web_search_tool_result"):
                # Defensive fallback: server-side tools are no longer used
                # (replaced by Tavily client tool), but convert to text if
                # the API ever returns them unexpectedly.
                t = _coerce_text(b.get("content", "") or b.get("input", "") or b.get("name", ""))
                if t:
                    assistant_content.append({"type": "text", "text": f"[server: {t[:2000]}]"})
            elif btype == "tool_use":
                tid = str(b.get("id", "")).strip()
                tname = str(b.get("name", "")).strip()
                tinput = b.get("input", {}) if isinstance(b.get("input", {}), dict) else {}
                if not tid or not tname:
                    logger.warning("Skipping malformed tool_use block: %s", b)
                    continue
                assistant_content.append({
                    "type": "tool_use",
                    "id": tid,
                    "name": tname,
                    "input": tinput,
                })
                tool_calls.append((tid, tname, tinput))
            else:
                # Preserve unknown future block types as text context.
                assistant_content.append({"type": "text", "text": _coerce_text(b)})

        return Turn(
            is_tool_round=True,
            text_parts=text_parts,
            tool_calls=tool_calls,
            finish_reason=str(stop_reason or "tool_use"),
            raw=response,
            extra={"assistant_content": assistant_content},
        )

    def append_assistant(self, msgs, turn):
        # Note: server_tool blocks are already converted to text in
        # parse_turn, so working messages never contain
        # server_tool_use/web_search_tool_result.
        msgs.append({"role": "assistant", "content": turn.extra["assistant_content"]})

    async def run_batch(self, batch, round_num):
        # Consecutive read-only tools run in parallel via execute_tools_batch;
        # everything else stays sequential. Resolved by bare name so test
        # patches on this module's execute_tools_batch keep working.
        return await execute_tools_batch(
            batch,
            self.tool_handlers,
            on_progress=self.on_progress,
            round_num=round_num,
            log_event=self.log_event,
            idempotency_cache=self.tool_execution_cache,
            tool_definitions=self.cached_tools,
        )

    def note_exec_results(self, round_num, exec_results):
        pass

    def append_tool_results(self, msgs, turn, exec_results, missing, warning_texts):
        tool_results = []
        for tid, _tname, _tinput, result, is_error in exec_results:
            tool_result_block = {
                "type": "tool_result",
                "tool_use_id": tid,
                "content": result,
            }
            if is_error:
                tool_result_block["is_error"] = True
            tool_results.append(tool_result_block)

        for tid, tname in missing:
            logger.warning("Safety net: missing tool_result for tool_use id=%s name=%s", tid, tname)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tid,
                "content": f"Tool execution skipped (internal error): no result was produced for {tname}",
                "is_error": True,
            })

        if tool_results:
            # Warnings must be appended AFTER tool_result blocks, not prepended.
            # Claude requires tool_use ids to have tool_result blocks immediately
            # after in the next user turn — a text block before them triggers
            # "tool_use ids without tool_result blocks immediately after".
            for text in warning_texts:
                tool_results.append({"type": "text", "text": text})
            msgs.append({"role": "user", "content": tool_results})
            return bool(warning_texts)
        if getattr(turn.raw, "stop_reason", None) == "pause_turn":
            msgs.append({"role": "user", "content": [{"type": "text", "text": "continue"}]})
        else:
            logger.warning(
                "No tool_results and not pause_turn (stop_reason=%s); appending fallback user message",
                getattr(turn.raw, "stop_reason", None),
            )
            msgs.append({"role": "user", "content": [{"type": "text", "text": "continue"}]})
        return False

    async def append_length_continuation(self, msgs, turn, partial_text, next_index, max_count):
        if partial_text:
            msgs.append({
                "role": "assistant",
                "content": [{"type": "text", "text": partial_text}],
            })
            _append_user_text_message(
                msgs,
                "Continue exactly from where the previous answer stopped. "
                "Do not restart, summarize, or repeat earlier text.",
            )
        else:
            # Truncation landed mid-tool_use: no text block survived, so there
            # is nothing to stitch — the whole round would otherwise be dropped
            # on the floor (2026-07-11 writer incident: a manuscript rewrite
            # inside one huge replace_in_manuscript call was lost this way).
            # Ask for a retry in smaller steps instead.
            _append_user_text_message(
                msgs,
                "Your previous response was cut off by the output-length "
                "limit before any usable content arrived — likely inside "
                "a large tool call. Redo that work in smaller steps: "
                "split big tool inputs into several smaller calls, and "
                "keep each call comfortably under the limit.",
            )
        return True

    async def on_truncated_final(self):
        pass

    # ── Forced-final phase ───────────────────────────────────────────

    def was_still_working(self, response):
        return getattr(response, "stop_reason", None) in ("tool_use", "pause_turn")

    def build_finalization_tools(self, finalization_tools):
        if not finalization_tools:
            return [], None
        allowed = set(finalization_tools)
        final_tool_names = [t["name"] for t in self.cached_tools if t.get("name") in allowed]
        if not final_tool_names:
            return [], None
        final_tools = [dict(t) for t in self.cached_tools if t.get("name") in set(final_tool_names)]
        # Preserve prompt caching semantics on the filtered list. Must
        # use the same 1h TTL as cached_system — Anthropic processes
        # `tools` before `system`, and a longer TTL cannot follow a
        # shorter one, so mixing 5m (default ephemeral) here with a 1h
        # system block raises a 400 (the diary task forced-final path
        # hit this).
        final_tools[-1] = {**final_tools[-1], "cache_control": _CACHE_CONTROL_1H}
        return final_tool_names, final_tools

    def append_user_text(self, msgs, text):
        _append_user_text_message(msgs, text)

    async def call_final(self, msgs, final_tools, limit_reason):
        create_kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": self.cached_system,
            "messages": _with_message_cache_breakpoint(msgs, self.message_cache_marks),
        }
        if final_tools:
            create_kwargs["tools"] = final_tools
        return await self._call(**create_kwargs)

    def parse_final(self, final, final_tool_names):
        if getattr(final, "stop_reason", None) == "max_tokens":
            logger.warning("Forced final response truncated by max_tokens (%d)", self.max_tokens)

        allowed = set(final_tool_names or [])
        final_assistant_content: list[dict] = []
        batch: list[tuple[str, str, dict]] = []
        text_parts: list[str] = []
        for block in final.content:
            b = _to_block_dict(block) or {"type": getattr(block, "type", "unknown")}
            btype = b.get("type")
            if btype == "text":
                replay_block = _content_block_for_replay(b)
                if replay_block:
                    final_assistant_content.append(replay_block)
                text_parts.append(str(b.get("text", "")))
            elif btype in _REPLAY_ONLY_BLOCK_TYPES:
                replay_block = _content_block_for_replay(b)
                if replay_block:
                    final_assistant_content.append(replay_block)
            elif btype == "tool_use":
                tid = str(b.get("id", "")).strip()
                tname = str(b.get("name", "")).strip()
                tinput = b.get("input", {}) if isinstance(b.get("input", {}), dict) else {}
                if tid and tname and tname in allowed:
                    final_assistant_content.append({
                        "type": "tool_use", "id": tid, "name": tname, "input": tinput,
                    })
                    batch.append((tid, tname, tinput))
                else:
                    logger.warning("Forced-final: ignoring non-finalization tool_use name=%s", tname)

        return FinalTurn(
            text_parts=text_parts,
            batch=batch,
            has_protocol=bool(batch),
            raw=final,
            extra={"assistant_content": final_assistant_content},
        )

    def append_final_assistant(self, msgs, final_turn):
        msgs.append({"role": "assistant", "content": final_turn.extra["assistant_content"]})

    def append_final_results(self, msgs, final_turn, exec_results):
        final_tool_results = []
        for tid, _tname, _tinput, result, is_error in exec_results:
            tr = {"type": "tool_result", "tool_use_id": tid, "content": result}
            if is_error:
                tr["is_error"] = True
            final_tool_results.append(tr)
        msgs.append({"role": "user", "content": final_tool_results})

    async def call_followup(self, msgs):
        # Cap output at 2K — closing remarks after a finalization tool don't
        # need the orchestrator's full max_tokens (typically 16K).
        return await self._call(
            model=self.model,
            max_tokens=min(self.max_tokens, 2048),
            system=self.cached_system,
            messages=_with_message_cache_breakpoint(msgs, self.message_cache_marks),  # no tools — force text
        )

    def extract_text_parts(self, response):
        return [b.text for b in response.content if b.type == "text"]

    async def recover_final_failure(self, msgs, err, limit_reason, was_still_working):
        # No protocol-strip recovery on the Anthropic path: this loop's own
        # parsing guarantees id/name correctness and 1:1 pairing, so a failed
        # forced-final call surfaces unchanged.
        raise err

    # ── Results ──────────────────────────────────────────────────────

    def update_tracker(self, tracker, *, rounds_used, was_interrupted, tool_work_details, response):
        _update_budget_tracker(
            tracker,
            total_cost=self.state.total_cost,
            rounds_used=rounds_used,
            was_interrupted=was_interrupted,
            tool_work_details=tool_work_details,
            response=response,
            model=self.model,
        )

    def make_result(self, parts, **_meta):
        return "\n".join(parts) if parts else EMPTY_RESPONSE_FALLBACK


async def chat_with_tools(
    messages: list[dict],
    *,
    client,
    model: str,
    tools: list[dict],
    tool_handlers: dict,
    system_prompt: str | list[dict],
    max_rounds: int = 50,
    max_tokens: int = 4096,
    max_input_tokens: int | None = None,
    recover_input_via_tools: bool = False,
    log_event=None,
    budget_usd: float = 0.30,
    budget_tracker: dict | None = None,
    on_progress=None,
    task_id: int | None = None,
    agent_name: str = "agent",
    mission_id: int | None = None,
    finalization_tools: list[str] | None = None,
    terminal_tools: list[str] | None = None,
    continue_on_length: bool = False,
    max_length_continuations: int = 1,
    thinking: dict | None = None,
    output_config: dict | None = None,
    provider_idle_timeout_sec: float | None = None,
) -> str:
    """Call Claude with tools, execute tool calls, loop until text response.

    Control flow is the shared engine in agent_loop.run_tool_loop; this module
    contributes the Anthropic protocol adapter.

    Args:
        messages: Conversation history.
        client: Anthropic AsyncAnthropic client.
        model: Model ID string.
        tools: Tool definitions (Anthropic API format).
        tool_handlers: Dict mapping tool name → async handler function.
        system_prompt: System prompt text or Anthropic system content blocks.
        max_rounds: Max tool-use rounds before forcing response.
        max_tokens: Max tokens for one response.
        max_input_tokens: Estimated request-input ceiling.
        recover_input_via_tools: Replace old large tool results with replay instructions before failing the ceiling.
        log_event: Optional callable(level, source, message, detail=None, task_id=None)
            for persistent error logging.
        budget_usd: Maximum USD budget for this call (default 0.30).
        budget_tracker: Optional dict — filled with {"total_cost", "rounds_used"} after return.
        on_progress: Optional async callable(event: str, detail: str) for live progress.
            Events: "thinking" (model's intermediate text), "tool_call" (tool invoked),
            "tool_result" (tool finished), "budget" (budget status update).
    """
    adapter = _ClaudeProtocolAdapter(
        client=client,
        model=model,
        tools=tools,
        tool_handlers=tool_handlers,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        max_input_tokens=max_input_tokens,
        recover_input_via_tools=recover_input_via_tools,
        on_progress=on_progress,
        log_event=log_event,
        thinking=thinking,
        output_config=output_config,
        provider_idle_timeout_sec=provider_idle_timeout_sec,
    )
    return await run_tool_loop(
        adapter,
        messages,
        max_rounds=max_rounds,
        max_tokens=max_tokens,
        budget_usd=budget_usd,
        budget_tracker=budget_tracker,
        on_progress=on_progress,
        task_id=task_id,
        log_event=log_event,
        agent_name=agent_name,
        mission_id=mission_id,
        finalization_tools=finalization_tools,
        terminal_tools=terminal_tools,
        continue_on_length=continue_on_length,
        max_length_continuations=max_length_continuations,
    )
